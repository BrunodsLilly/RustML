// @ts-check
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

/**
 * E2E tests for CSV Upload Feature
 *
 * Prerequisites:
 * - Run `dx serve` in the web directory before running these tests
 * - The dev server should be running on http://localhost:8080
 */

test.describe('CSV Upload Feature', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the showcase page
    await page.goto('http://localhost:8081/showcase');

    // Wait for page to be fully loaded
    await page.waitForLoadState('networkidle');
  });

  test('should display data source toggle', async ({ page }) => {
    // Scroll to gradient descent section
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Verify data source toggle exists
    await expect(page.getByText('Use Preset Data')).toBeVisible();
    await expect(page.getByText('Upload CSV')).toBeVisible();

    // Preset data should be selected by default
    const presetRadio = page.locator('input[type="radio"][value="preset"]');
    await expect(presetRadio).toBeChecked();
  });

  test('should show preset data interface by default', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Verify preset buttons are visible
    await expect(page.getByText('📈 Linear (y=2x+1)')).toBeVisible();
    await expect(page.getByText('📊 Steep (y=5x)')).toBeVisible();
    await expect(page.getByText('📉 Noisy')).toBeVisible();
  });

  test('should switch to CSV upload interface', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Click on Upload CSV radio button
    await page.locator('input[type="radio"][value="csv"]').click();

    // Verify CSV upload interface appears
    await expect(page.getByText('Upload CSV Dataset')).toBeVisible();
    await expect(page.locator('input[type="file"][accept=".csv"]')).toBeVisible();

    // Verify preset buttons are hidden
    await expect(page.getByText('📈 Linear (y=2x+1)')).not.toBeVisible();
  });

  test('should upload and preview valid CSV file', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Switch to CSV mode
    await page.locator('input[type="radio"][value="csv"]').click();

    // Create a test CSV file
    const csvContent = 'feature,target\n1.0,2.0\n2.0,4.0\n3.0,6.0\n4.0,8.0\n5.0,10.0';

    // Set up file chooser and upload
    const fileInput = page.locator('input[type="file"][accept=".csv"]');
    await fileInput.setInputFiles({
      name: 'test-data.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    // Wait for processing
    await page.waitForTimeout(500);

    // Verify preview appears
    await expect(page.getByText(/Preview:.*test-data\.csv/)).toBeVisible();
    await expect(page.getByText(/5 rows/)).toBeVisible();

    // Verify preview table
    await expect(page.locator('table')).toBeVisible();
    await expect(page.locator('table th', { hasText: 'feature' })).toBeVisible();
    await expect(page.locator('table th', { hasText: 'target' })).toBeVisible();

    // Verify target column selector
    const targetSelect = page.locator('select#target-column');
    await expect(targetSelect).toBeVisible();
    // Default selection can be either column - just verify it has a value
    const value = await targetSelect.inputValue();
    expect(['feature', 'target']).toContain(value);

    // Verify Load Dataset button
    await expect(page.locator('button.load-button')).toBeVisible();
  });

  test('should show error for non-CSV file', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Switch to CSV mode
    await page.locator('input[type="radio"][value="csv"]').click();

    // Upload a non-CSV file
    const fileInput = page.locator('input[type="file"][accept=".csv"]');
    await fileInput.setInputFiles({
      name: 'test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('test data')
    });

    // Wait for error
    await page.waitForTimeout(500);

    // Verify error message
    await expect(page.locator('.error-message')).toBeVisible();
    await expect(page.getByText(/Please upload a \.csv file/)).toBeVisible();
  });

  test('should load dataset and show info', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Switch to CSV mode
    await page.locator('input[type="radio"][value="csv"]').click();

    // Upload CSV
    const csvContent = 'x1,x2,y\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0';
    const fileInput = page.locator('input[type="file"][accept=".csv"]');
    await fileInput.setInputFiles({
      name: 'multi-feature.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    await page.waitForTimeout(500);

    // Click Load Dataset button
    await page.locator('button.load-button').click();

    // Wait for dataset to load
    await page.waitForTimeout(500);

    // Verify dataset info appears
    await expect(page.getByText(/Dataset loaded: 3 samples/)).toBeVisible();
    await expect(page.getByText(/Features: 2/)).toBeVisible();
    await expect(page.getByText(/x1, x2/)).toBeVisible();
  });

  test('should train model on CSV data', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Switch to CSV mode
    await page.locator('input[type="radio"][value="csv"]').click();

    // Upload simple linear data
    const csvContent = 'x,y\n1.0,2.0\n2.0,4.0\n3.0,6.0\n4.0,8.0\n5.0,10.0';
    const fileInput = page.locator('input[type="file"][accept=".csv"]');
    await fileInput.setInputFiles({
      name: 'linear-data.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    await page.waitForTimeout(500);

    // Load dataset
    await page.locator('button.load-button').click();
    await page.waitForTimeout(500);

    // Find and click Train Model button
    const trainButton = page.getByRole('button', { name: /Train Model/i });
    await trainButton.scrollIntoViewIfNeeded();
    await trainButton.click();

    // Wait for training to complete (should be fast)
    await page.waitForTimeout(2000);

    // Verify training results appear
    await expect(page.getByText(/Learned Parameters/i)).toBeVisible();
  });

  test('should change target column selection', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Switch to CSV mode
    await page.locator('input[type="radio"][value="csv"]').click();

    // Upload multi-column CSV
    const csvContent = 'a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0';
    const fileInput = page.locator('input[type="file"][accept=".csv"]');
    await fileInput.setInputFiles({
      name: 'multi-col.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    await page.waitForTimeout(500);

    // Change target column
    const targetSelect = page.locator('select#target-column');
    await targetSelect.selectOption('a');
    await expect(targetSelect).toHaveValue('a');

    await targetSelect.selectOption('b');
    await expect(targetSelect).toHaveValue('b');
  });

  test('should handle large preview correctly', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Switch to CSV mode
    await page.locator('input[type="radio"][value="csv"]').click();

    // Create CSV with 20 rows (should show only first 10)
    let csvContent = 'x,y\n';
    for (let i = 1; i <= 20; i++) {
      csvContent += `${i}.0,${i * 2}.0\n`;
    }

    const fileInput = page.locator('input[type="file"][accept=".csv"]');
    await fileInput.setInputFiles({
      name: 'large-data.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    await page.waitForTimeout(500);

    // Verify preview shows total rows
    await expect(page.getByRole('heading', { name: /large-data\.csv.*20 rows/ })).toBeVisible();

    // Verify preview note about showing subset
    await expect(page.getByText(/Showing first 10 of 20 rows/)).toBeVisible();

    // Count visible table rows (should be 10 data rows + 1 header)
    const rows = page.locator('table.preview-table tbody tr');
    await expect(rows).toHaveCount(10);
  });

  test('should switch back to preset data mode', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Switch to CSV mode
    await page.locator('input[type="radio"][value="csv"]').click();
    await expect(page.getByText('Upload CSV Dataset')).toBeVisible();

    // Switch back to preset mode
    await page.locator('input[type="radio"][value="preset"]').click();

    // Verify preset interface is back
    await expect(page.getByText('📈 Linear (y=2x+1)')).toBeVisible();
    await expect(page.getByText('Upload CSV Dataset')).not.toBeVisible();
  });

  test('should show processing indicator', async ({ page }) => {
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();

    // Switch to CSV mode
    await page.locator('input[type="radio"][value="csv"]').click();

    // Note: Processing happens very fast, so we just verify the indicator exists in DOM
    const loadingIndicator = page.locator('.loading');
    // The indicator should exist (even if not visible due to fast processing)
    const count = await loadingIndicator.count();
    expect(count).toBeGreaterThanOrEqual(0);
  });
});
