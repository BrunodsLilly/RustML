# Testing the ML Library Showcase Web App

This document describes how to run end-to-end tests for the Dioxus web application.

## Prerequisites

1. **Node.js and npm** - Required to run Playwright tests
2. **Dioxus CLI** - Required to run the dev server (`cargo install dioxus-cli`)
3. **Playwright** - Will be installed via npm

## Setup

### 1. Install Dependencies

```bash
cd web
npm install
npx playwright install
```

This will install:
- Playwright test framework
- Browser binaries (Chromium, Firefox, WebKit)

### 2. Start the Development Server

In a separate terminal, start the Dioxus dev server:

```bash
cd web
dx serve
```

Wait for the server to start (usually at `http://localhost:8080`).

## Running Tests

### Run All Tests

```bash
npm test
```

This runs all tests in headless mode across all configured browsers.

### Run Tests in Headed Mode

To see the browser while tests run:

```bash
npm run test:headed
```

### Run Tests in Debug Mode

To debug tests step-by-step:

```bash
npm run test:debug
```

### Run Specific Test Suite

To run only the showcase tests:

```bash
npm run test:showcase
```

### Run Tests in UI Mode

Playwright has an interactive UI mode for exploring tests:

```bash
npm run test:ui
```

## Test Coverage

Current test coverage for the showcase page:

### Navigation Tests
- ✅ Main page displays showcase link
- ✅ Clicking showcase link navigates to `/showcase`
- ✅ Back button returns to main page

### Static Demonstrations
- ✅ Vector addition demo displays correctly
- ✅ Scalar multiplication demo displays correctly
- ✅ Dot product demo displays correctly
- ✅ All static results are accurate

### Interactive Calculator
- ✅ Calculator inputs are visible and interactive
- ✅ Default values display correct results
- ✅ Changing Vector A updates all calculations
- ✅ Changing Vector B updates relevant calculations
- ✅ Changing scalar updates scalar multiplication
- ✅ Decimal inputs work correctly

### Styling
- ✅ Demo sections have proper CSS classes
- ✅ Interactive section is styled differently
- ✅ Results section displays correctly

## Test Structure

```
web/
├── tests/
│   └── showcase.spec.js       # Main showcase test suite
├── package.json                # npm configuration with test scripts
├── playwright.config.js        # Playwright configuration
└── TESTING.md                  # This file
```

## Writing New Tests

To add new tests:

1. Create a new `.spec.js` file in the `tests/` directory
2. Import Playwright test utilities:
   ```javascript
   const { test, expect } = require('@playwright/test');
   ```
3. Write test suites using `test.describe()` and individual tests using `test()`
4. Use Playwright's assertions with `expect()`

Example:

```javascript
test('should do something', async ({ page }) => {
  await page.goto('http://localhost:8080/your-route');
  await expect(page.getByText('Some Text')).toBeVisible();
});
```

## Test Reports

After running tests, Playwright generates an HTML report:

```bash
npx playwright show-report
```

This opens a detailed report in your browser showing:
- Test results for each browser
- Screenshots of failures
- Traces for debugging
- Performance metrics

## CI/CD Integration

The tests are configured to work in CI environments:

- Retries failed tests twice in CI
- Runs tests sequentially in CI (not parallel)
- Captures traces and screenshots on failure

To run in CI mode:

```bash
CI=true npm test
```

## Troubleshooting

### Server Not Running

If tests fail with connection errors:

```
Error: page.goto: net::ERR_CONNECTION_REFUSED
```

Make sure `dx serve` is running in another terminal.

### Port Already in Use

If port 8080 is busy, you can change it:

1. In `dx serve --port <PORT>`
2. Update `baseURL` in `playwright.config.js`
3. Update test URLs if hardcoded

### Browser Installation Issues

If browsers aren't installed:

```bash
npx playwright install
```

To install specific browsers:

```bash
npx playwright install chromium
npx playwright install firefox
npx playwright install webkit
```

## Future Test Ideas

- [ ] Test routing between all pages
- [ ] Test data persistence in calculator
- [ ] Test error handling for invalid inputs
- [ ] Test mobile responsiveness
- [ ] Test accessibility (a11y)
- [ ] Test keyboard navigation
- [ ] Visual regression tests
- [ ] Performance tests
- [ ] Add tests for linear regression when integrated
- [ ] Add tests for plotting features when integrated

## Resources

- [Playwright Documentation](https://playwright.dev/)
- [Dioxus Testing Guide](https://dioxuslabs.com/learn/0.6/testing)
- [Best Practices for E2E Testing](https://playwright.dev/docs/best-practices)
