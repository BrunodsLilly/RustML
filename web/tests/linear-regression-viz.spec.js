// @ts-check
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

/**
 * E2E tests for Linear Regression Visualizer
 *
 * Prerequisites:
 * - Run `dx serve` in the web directory before running these tests
 * - The dev server should be running on http://localhost:8080
 *
 * Tests cover:
 * - CSV upload → training → visualization display
 * - Tab navigation (Coefficients, Importance, Correlations)
 * - Component rendering and interactivity
 */

// Helper function to create test CSV file
function createTestCSV(filename, content) {
  const testDir = path.join(__dirname, 'fixtures');
  if (!fs.existsSync(testDir)) {
    fs.mkdirSync(testDir, { recursive: true });
  }
  const filepath = path.join(testDir, filename);
  fs.writeFileSync(filepath, content);
  return filepath;
}

test.describe('Linear Regression Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the showcase page
    await page.goto('http://localhost:8080/showcase');
    await page.waitForLoadState('networkidle');

    // Scroll to gradient descent section
    const section = page.locator('section.gradient-descent');
    await section.scrollIntoViewIfNeeded();
  });

  test('should display visualizer after CSV training', async ({ page }) => {
    // Create test CSV with multiple features
    const csvContent = `feature1,feature2,feature3,target
1.0,2.0,3.0,10.5
2.0,3.0,4.0,15.7
3.0,4.0,5.0,20.9
4.0,5.0,6.0,26.1
5.0,6.0,7.0,31.3`;

    const csvPath = createTestCSV('multi_feature.csv', csvContent);

    // Switch to CSV upload
    await page.getByText('Upload CSV').click();

    // Upload file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(csvPath);

    // Wait for preview to load
    await expect(page.getByText('Dataset loaded')).toBeVisible({ timeout: 5000 });

    // Select target column
    const targetSelect = page.locator('select[name="target-column"]');
    await targetSelect.selectOption('target');

    // Confirm selection
    await page.getByText('Use This Data').click();

    // Train model
    await page.getByText('🚀 Train Model').click();

    // Wait for training to complete
    await page.waitForTimeout(2000);

    // Verify visualizer appears
    await expect(page.locator('.linear-regression-visualizer')).toBeVisible({ timeout: 5000 });

    // Verify header
    await expect(page.getByText('📊 Model Analysis & Insights')).toBeVisible();

    // Cleanup
    fs.unlinkSync(csvPath);
  });

  test('should have three tabs: Coefficients, Importance, Correlations', async ({ page }) => {
    // Create and upload CSV
    const csvContent = `x1,x2,y
1.0,2.0,5.0
2.0,3.0,8.0
3.0,4.0,11.0`;

    const csvPath = createTestCSV('simple.csv', csvContent);

    await page.getByText('Upload CSV').click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(csvPath);

    await page.waitForTimeout(1000);
    const targetSelect = page.locator('select[name="target-column"]');
    await targetSelect.selectOption('y');
    await page.getByText('Use This Data').click();
    await page.getByText('🚀 Train Model').click();
    await page.waitForTimeout(2000);

    // Verify tabs exist
    await expect(page.getByText('📋 Coefficients')).toBeVisible();
    await expect(page.getByText('⭐ Importance')).toBeVisible();
    await expect(page.getByText('🔥 Correlations')).toBeVisible();

    // Cleanup
    fs.unlinkSync(csvPath);
  });

  test('should display coefficients table with feature names', async ({ page }) => {
    const csvContent = `age,income,score
25,50000,75
30,60000,85
35,70000,90`;

    const csvPath = createTestCSV('coefficients.csv', csvContent);

    await page.getByText('Upload CSV').click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(csvPath);
    await page.waitForTimeout(1000);

    const targetSelect = page.locator('select[name="target-column"]');
    await targetSelect.selectOption('score');
    await page.getByText('Use This Data').click();
    await page.getByText('🚀 Train Model').click();
    await page.waitForTimeout(2000);

    // Click Coefficients tab (should be active by default)
    await page.getByText('📋 Coefficients').click();

    // Verify coefficient display header
    await expect(page.getByText('📊 Model Coefficients')).toBeVisible();

    // Verify feature names in table
    await expect(page.getByText('age')).toBeVisible();
    await expect(page.getByText('income')).toBeVisible();

    // Verify table structure
    const table = page.locator('.coefficients-table table');
    await expect(table).toBeVisible();

    // Verify columns exist
    await expect(page.locator('th:has-text("Feature")')).toBeVisible();
    await expect(page.locator('th:has-text("Weight")')).toBeVisible();

    // Cleanup
    fs.unlinkSync(csvPath);
  });

  test('should switch between tabs correctly', async ({ page }) => {
    const csvContent = `x1,x2,target
1.0,2.0,3.0
2.0,4.0,6.0
3.0,6.0,9.0`;

    const csvPath = createTestCSV('tabs.csv', csvContent);

    await page.getByText('Upload CSV').click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(csvPath);
    await page.waitForTimeout(1000);

    const targetSelect = page.locator('select[name="target-column"]');
    await targetSelect.selectOption('target');
    await page.getByText('Use This Data').click();
    await page.getByText('🚀 Train Model').click();
    await page.waitForTimeout(2000);

    // Start on Coefficients tab
    await expect(page.locator('.tab-button.active')).toContainText('Coefficients');
    await expect(page.locator('.coefficient-display')).toBeVisible();

    // Switch to Importance tab
    await page.getByText('⭐ Importance').click();
    await page.waitForTimeout(500);
    await expect(page.locator('.tab-button.active')).toContainText('Importance');
    await expect(page.locator('.feature-importance-chart')).toBeVisible();

    // Switch to Correlations tab
    await page.getByText('🔥 Correlations').click();
    await page.waitForTimeout(500);
    await expect(page.locator('.tab-button.active')).toContainText('Correlations');
    await expect(page.locator('.correlation-heatmap')).toBeVisible();

    // Switch back to Coefficients
    await page.getByText('📋 Coefficients').click();
    await page.waitForTimeout(500);
    await expect(page.locator('.coefficient-display')).toBeVisible();

    // Cleanup
    fs.unlinkSync(csvPath);
  });

  test('should display feature importance chart with bars', async ({ page }) => {
    const csvContent = `height,weight,age,bmi
160,60,25,23.4
170,70,30,24.2
180,80,35,24.7`;

    const csvPath = createTestCSV('importance.csv', csvContent);

    await page.getByText('Upload CSV').click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(csvPath);
    await page.waitForTimeout(1000);

    const targetSelect = page.locator('select[name="target-column"]');
    await targetSelect.selectOption('bmi');
    await page.getByText('Use This Data').click();
    await page.getByText('🚀 Train Model').click();
    await page.waitForTimeout(2000);

    // Switch to Importance tab
    await page.getByText('⭐ Importance').click();

    // Verify importance chart header
    await expect(page.getByText('⭐ Feature Importance')).toBeVisible();

    // Verify sort controls exist
    await expect(page.getByText('Sort by:')).toBeVisible();

    // Verify importance bars container
    await expect(page.locator('.importance-bars')).toBeVisible();

    // Verify at least one importance row is displayed
    await expect(page.locator('.importance-row').first()).toBeVisible();

    // Cleanup
    fs.unlinkSync(csvPath);
  });

  test('should display correlation heatmap with SVG grid', async ({ page }) => {
    const csvContent = `a,b,c,target
1,2,3,6
2,4,6,12
3,6,9,18`;

    const csvPath = createTestCSV('correlation.csv', csvContent);

    await page.getByText('Upload CSV').click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(csvPath);
    await page.waitForTimeout(1000);

    const targetSelect = page.locator('select[name="target-column"]');
    await targetSelect.selectOption('target');
    await page.getByText('Use This Data').click();
    await page.getByText('🚀 Train Model').click();
    await page.waitForTimeout(2000);

    // Switch to Correlations tab
    await page.getByText('🔥 Correlations').click();

    // Verify correlation heatmap header
    await expect(page.getByText('🔥 Feature Correlation Matrix')).toBeVisible();

    // Verify SVG heatmap exists
    const heatmap = page.locator('.heatmap-container svg');
    await expect(heatmap).toBeVisible();

    // Verify correlation insights section
    await expect(page.getByText('Key Insights')).toBeVisible();

    // Cleanup
    fs.unlinkSync(csvPath);
  });

  test('should display model performance summary', async ({ page }) => {
    const csvContent = `x,y
1,2
2,4
3,6`;

    const csvPath = createTestCSV('performance.csv', csvContent);

    await page.getByText('Upload CSV').click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(csvPath);
    await page.waitForTimeout(1000);

    const targetSelect = page.locator('select[name="target-column"]');
    await targetSelect.selectOption('y');
    await page.getByText('Use This Data').click();
    await page.getByText('🚀 Train Model').click();
    await page.waitForTimeout(2000);

    // Verify performance summary section
    await expect(page.getByText('Model Performance')).toBeVisible();

    // Verify performance metrics exist
    await expect(page.getByText('Final Cost')).toBeVisible();
    await expect(page.getByText('Training Iterations')).toBeVisible();
    await expect(page.getByText('Cost Reduction')).toBeVisible();

    // Cleanup
    fs.unlinkSync(csvPath);
  });

  test('should show contextual tips', async ({ page }) => {
    const csvContent = `feature1,feature2,target
1,1,2
2,2,4
3,3,6`;

    const csvPath = createTestCSV('tips.csv', csvContent);

    await page.getByText('Upload CSV').click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(csvPath);
    await page.waitForTimeout(1000);

    const targetSelect = page.locator('select[name="target-column"]');
    await targetSelect.selectOption('target');
    await page.getByText('Use This Data').click();
    await page.getByText('🚀 Train Model').click();
    await page.waitForTimeout(2000);

    // Verify tips panel exists
    await expect(page.getByText('💡 Tips')).toBeVisible();

    // Tips panel should have at least one tip
    const tipsPanel = page.locator('.tips-panel ul');
    await expect(tipsPanel).toBeVisible();

    // Cleanup
    fs.unlinkSync(csvPath);
  });

  test('should handle single-feature dataset (fallback to scatter plot)', async ({ page }) => {
    // Use preset data which is single-feature
    await page.getByText('📈 Linear (y=2x+1)').click();

    // Train model
    await page.getByText('🚀 Train Model').click();
    await page.waitForTimeout(2000);

    // Should show scatter plot, NOT visualizer
    await expect(page.locator('.scatter-plot-container')).toBeVisible();
    await expect(page.locator('.linear-regression-visualizer')).not.toBeVisible();
  });
});

test.afterAll(() => {
  // Clean up fixtures directory
  const testDir = path.join(__dirname, 'fixtures');
  if (fs.existsSync(testDir)) {
    fs.rmSync(testDir, { recursive: true, force: true });
  }
});
