// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * E2E tests for the ML Library Showcase
 *
 * Prerequisites:
 * - Run `dx serve` in the web directory before running these tests
 * - The dev server should be running on http://localhost:8080
 */

test.describe('ML Library Showcase', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the main page
    await page.goto('http://localhost:8080');
  });

  test('should display main page with showcase link', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle("Bruno's Web App");

    // Verify showcase link exists
    const showcaseLink = page.getByRole('link', { name: 'ML Library Showcase' });
    await expect(showcaseLink).toBeVisible();
  });

  test('should navigate to showcase page', async ({ page }) => {
    // Click on showcase link
    await page.getByRole('link', { name: 'ML Library Showcase' }).click();

    // Verify we're on the showcase page
    await expect(page).toHaveURL('http://localhost:8080/showcase');

    // Verify page heading
    const heading = page.getByRole('heading', { name: 'ML Library Showcase', level: 1 });
    await expect(heading).toBeVisible();
  });

  test('should display static vector operation demos', async ({ page }) => {
    await page.goto('http://localhost:8080/showcase');

    // Check Vector Addition demo
    await expect(page.getByText('Vector Addition')).toBeVisible();
    await expect(page.getByText('v![1, 2, 3] + v![4, 5, 6]')).toBeVisible();
    await expect(page.getByText('Result: [5, 7, 9]')).toBeVisible();

    // Check Scalar Multiplication demo
    await expect(page.getByText('Vector Scalar Multiplication')).toBeVisible();
    await expect(page.getByText('v![1, 2, 3] * 10')).toBeVisible();
    await expect(page.getByText('Result: [10, 20, 30]')).toBeVisible();

    // Check Dot Product demo
    await expect(page.getByText('Dot Product')).toBeVisible();
    await expect(page.getByText('v![1, 2, 3].dot(&v![4, 5, 6])')).toBeVisible();
    await expect(page.getByText('Result: 32')).toBeVisible();
  });

  test('should display interactive vector calculator', async ({ page }) => {
    await page.goto('http://localhost:8080/showcase');

    // Check interactive calculator section
    await expect(page.getByRole('heading', { name: 'Interactive Vector Calculator', level: 2 })).toBeVisible();

    // Verify input labels
    await expect(page.getByText('Vector A:')).toBeVisible();
    await expect(page.getByText('Vector B:')).toBeVisible();
    await expect(page.getByText('Scalar:')).toBeVisible();

    // Verify results section
    await expect(page.getByRole('heading', { name: 'Results:', level: 3 })).toBeVisible();
    await expect(page.getByText('A + B =')).toBeVisible();
    await expect(page.getByText('A · B =')).toBeVisible();
    await expect(page.getByText('A × scalar =')).toBeVisible();
  });

  test('should calculate correct results with default values', async ({ page }) => {
    await page.goto('http://localhost:8080/showcase');

    // Check default results (Vector A: [1,2,3], Vector B: [4,5,6], Scalar: 2)
    const additionResult = page.locator('code', { hasText: '[5.0, 7.0, 9.0]' });
    const dotProductResult = page.locator('code', { hasText: '32' });
    const scalarResult = page.locator('code', { hasText: '[2.0, 4.0, 6.0]' });

    await expect(additionResult).toBeVisible();
    await expect(dotProductResult).toBeVisible();
    await expect(scalarResult).toBeVisible();
  });

  test('should update results when Vector A changes', async ({ page }) => {
    await page.goto('http://localhost:8080/showcase');

    // Get the first input for Vector A
    const vectorAInput = page.getByRole('spinbutton').first();

    // Change Vector A[0] from 1 to 5
    await vectorAInput.fill('5');

    // Wait a bit for reactivity
    await page.waitForTimeout(100);

    // Verify updated results
    // A + B = [5,2,3] + [4,5,6] = [9,7,9]
    await expect(page.locator('code', { hasText: '[9.0, 7.0, 9.0]' })).toBeVisible();

    // A · B = [5,2,3] · [4,5,6] = 20 + 10 + 18 = 48
    await expect(page.locator('code', { hasText: '48' })).toBeVisible();

    // A × scalar = [5,2,3] × 2 = [10,4,6]
    await expect(page.locator('code', { hasText: '[10.0, 4.0, 6.0]' })).toBeVisible();
  });

  test('should update results when scalar changes', async ({ page }) => {
    await page.goto('http://localhost:8080/showcase');

    // Get the scalar input
    const scalarInput = page.locator('.input-group > input');

    // Change scalar from 2 to 3
    await scalarInput.fill('3');

    // Wait a bit for reactivity
    await page.waitForTimeout(100);

    // Verify updated scalar multiplication result
    // A × scalar = [1,2,3] × 3 = [3,6,9]
    await expect(page.locator('code', { hasText: '[3.0, 6.0, 9.0]' })).toBeVisible();

    // Other results should remain the same
    await expect(page.locator('code', { hasText: '[5.0, 7.0, 9.0]' })).toBeVisible();
    await expect(page.locator('code', { hasText: '32' })).toBeVisible();
  });

  test('should handle decimal inputs', async ({ page }) => {
    await page.goto('http://localhost:8080/showcase');

    // Change Vector A[0] to 1.5
    const vectorAInput = page.getByRole('spinbutton').first();
    await vectorAInput.fill('1.5');

    // Change Vector B[0] to 2.5
    const vectorBInput = page.getByRole('spinbutton').nth(3);
    await vectorBInput.fill('2.5');

    // Wait for reactivity
    await page.waitForTimeout(100);

    // A + B should have 4.0 as first element (1.5 + 2.5)
    await expect(page.locator('code', { hasText: '[4.0, 7.0, 9.0]' })).toBeVisible();
  });

  test('should navigate back to main page', async ({ page }) => {
    await page.goto('http://localhost:8080/showcase');

    // Click back to main link
    await page.getByRole('link', { name: '← Back to Main' }).click();

    // Verify we're back on the main page
    await expect(page).toHaveURL('http://localhost:8080/');
    await expect(page.getByRole('heading', { name: 'Bruno', level: 1 })).toBeVisible();
  });

  test('should have proper CSS styling', async ({ page }) => {
    await page.goto('http://localhost:8080/showcase');

    // Check that demo sections have proper styling
    const demoSection = page.locator('.demo-section').first();
    await expect(demoSection).toBeVisible();

    // Check that interactive section exists
    const interactiveSection = page.locator('.interactive');
    await expect(interactiveSection).toBeVisible();

    // Check that results section exists
    const resultsSection = page.locator('.results');
    await expect(resultsSection).toBeVisible();
  });
});
