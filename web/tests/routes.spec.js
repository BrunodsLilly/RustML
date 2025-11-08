// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * E2E tests for all application routes
 *
 * Prerequisites:
 * - Run `dx serve` in the web directory before running these tests
 * - The dev server should be running on http://localhost:8080
 */

test.describe('Application Routes', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the main page before each test
    await page.goto('http://localhost:8080');
  });

  test.describe('Main View (/) Route', () => {
    test('should display main page with correct title', async ({ page }) => {
      // Verify we're on the main page
      await expect(page).toHaveURL('http://localhost:8080/');

      // Check page title
      await expect(page).toHaveTitle("Bruno's Web App");

      // Verify main heading
      const heading = page.getByRole('heading', { name: 'Bruno', level: 1 });
      await expect(heading).toBeVisible();
    });

    test('should display all navigation links', async ({ page }) => {
      // Verify all expected links are present
      await expect(page.getByRole('link', { name: 'ML Library Showcase' })).toBeVisible();
      await expect(page.getByRole('link', { name: 'Optimizer Visualizer' })).toBeVisible();
      await expect(page.getByRole('link', { name: 'Coursera ML Course' })).toBeVisible();
    });
  });

  test.describe('Showcase View (/showcase) Route', () => {
    test('should navigate to showcase page', async ({ page }) => {
      // Click on showcase link
      await page.getByRole('link', { name: 'ML Library Showcase' }).click();

      // Verify we're on the showcase page
      await expect(page).toHaveURL('http://localhost:8080/showcase');

      // Verify page heading
      const heading = page.getByRole('heading', { name: 'ML Library Showcase', level: 1 });
      await expect(heading).toBeVisible();
    });

    test('should display interactive vector calculator', async ({ page }) => {
      await page.goto('http://localhost:8080/showcase');

      // Check interactive calculator section
      await expect(page.getByRole('heading', { name: 'Interactive Vector Calculator', level: 2 })).toBeVisible();

      // Verify input labels
      await expect(page.getByText('Vector A:')).toBeVisible();
      await expect(page.getByText('Vector B:')).toBeVisible();
      await expect(page.getByText('Scalar:')).toBeVisible();
    });

    test('should navigate back to main page from showcase', async ({ page }) => {
      await page.goto('http://localhost:8080/showcase');

      // Click back to main link
      await page.getByRole('link', { name: '← Back to Main' }).click();

      // Verify we're back on the main page
      await expect(page).toHaveURL('http://localhost:8080/');
      await expect(page.getByRole('heading', { name: 'Bruno', level: 1 })).toBeVisible();
    });
  });

  test.describe('Optimizers View (/optimizers) Route', () => {
    test('should navigate to optimizers page', async ({ page }) => {
      // Click on optimizer visualizer link
      await page.getByRole('link', { name: 'Optimizer Visualizer' }).click();

      // Verify we're on the optimizers page
      await expect(page).toHaveURL('http://localhost:8080/optimizers');

      // Verify page heading
      const heading = page.getByRole('heading', { name: '⚡ Optimizer Race Track', level: 1 });
      await expect(heading).toBeVisible();
    });

    test('should display all loss function buttons', async ({ page }) => {
      await page.goto('http://localhost:8080/optimizers');

      // Verify all 6 loss functions are present
      await expect(page.getByRole('button', { name: /Quadratic Bowl/ })).toBeVisible();
      await expect(page.getByRole('button', { name: /Rosenbrock/ })).toBeVisible();
      await expect(page.getByRole('button', { name: /Saddle Point/ })).toBeVisible();
      await expect(page.getByRole('button', { name: /Beale/ })).toBeVisible();
      await expect(page.getByRole('button', { name: /Himmelblau/ })).toBeVisible();
      await expect(page.getByRole('button', { name: /Rastrigin/ })).toBeVisible();
    });

    test('should display control buttons', async ({ page }) => {
      await page.goto('http://localhost:8080/optimizers');

      // Verify control buttons
      await expect(page.getByRole('button', { name: '▶ Start Race' })).toBeVisible();
      await expect(page.getByRole('button', { name: '🔄 Reset' })).toBeVisible();

      // Verify speed slider
      await expect(page.getByRole('slider')).toBeVisible();

      // Verify checkbox for landscape
      await expect(page.getByRole('checkbox', { name: 'Show Loss Landscape' })).toBeVisible();
    });

    test('should display all 4 optimizer panels', async ({ page }) => {
      await page.goto('http://localhost:8080/optimizers');

      // Verify all 4 optimizers are displayed
      await expect(page.getByText('SGD').first()).toBeVisible();
      await expect(page.getByText('Momentum').first()).toBeVisible();
      await expect(page.getByText('RMSprop').first()).toBeVisible();
      await expect(page.getByText('Adam').first()).toBeVisible();
    });

    test('should display performance metrics', async ({ page }) => {
      await page.goto('http://localhost:8080/optimizers');

      // Verify WASM performance section
      await expect(page.getByRole('heading', { name: '⚡WASM Performance', level: 3 })).toBeVisible();

      // Verify metrics are displayed
      await expect(page.getByText(/Iterations\/sec/)).toBeVisible();
      await expect(page.getByText(/Frame time/)).toBeVisible();
      await expect(page.getByText(/Total computations/)).toBeVisible();
    });

    test('should display WASM indicator text', async ({ page }) => {
      await page.goto('http://localhost:8080/optimizers');

      // Verify WASM/Rust indicator
      await expect(page.getByText('🦀 Rust + WASM')).toBeVisible();
      await expect(page.getByText(/Running entirely in your browser with Rust \+ WASM/)).toBeVisible();
    });
  });

  test.describe('Courses View (/courses) Route', () => {
    test('should navigate to courses page', async ({ page }) => {
      // Click on Coursera ML Course link
      await page.getByRole('link', { name: 'Coursera ML Course' }).click();

      // Verify we're on the courses page
      await expect(page).toHaveURL('http://localhost:8080/courses');
    });

    test('should display courses content', async ({ page }) => {
      await page.goto('http://localhost:8080/courses');

      // Verify courses text is displayed
      await expect(page.getByText('courses')).toBeVisible();
    });

    test('should have link back to main page', async ({ page }) => {
      await page.goto('http://localhost:8080/courses');

      // Verify link to main page exists
      await expect(page.getByRole('link', { name: 'Main' })).toBeVisible();
    });
  });

  test.describe('Route Navigation', () => {
    test('should navigate between all routes without errors', async ({ page }) => {
      // Start at main page
      await expect(page).toHaveURL('http://localhost:8080/');

      // Navigate to showcase
      await page.getByRole('link', { name: 'ML Library Showcase' }).click();
      await expect(page).toHaveURL('http://localhost:8080/showcase');

      // Back to main
      await page.getByRole('link', { name: '← Back to Main' }).click();
      await expect(page).toHaveURL('http://localhost:8080/');

      // Navigate to optimizers
      await page.getByRole('link', { name: 'Optimizer Visualizer' }).click();
      await expect(page).toHaveURL('http://localhost:8080/optimizers');

      // Navigate to courses via direct URL
      await page.goto('http://localhost:8080/courses');
      await expect(page).toHaveURL('http://localhost:8080/courses');

      // Back to main from courses
      await page.getByRole('link', { name: 'Main' }).click();
      await expect(page).toHaveURL('http://localhost:8080/');
    });

    test('should handle direct URL navigation to all routes', async ({ page }) => {
      // Test direct navigation to each route
      const routes = [
        { url: 'http://localhost:8080/', heading: 'Bruno' },
        { url: 'http://localhost:8080/showcase', heading: 'ML Library Showcase' },
        { url: 'http://localhost:8080/optimizers', heading: '⚡ Optimizer Race Track' },
      ];

      for (const route of routes) {
        await page.goto(route.url);
        await expect(page).toHaveURL(route.url);
        await expect(page.getByRole('heading', { name: route.heading, level: 1 })).toBeVisible();
      }
    });
  });
});
