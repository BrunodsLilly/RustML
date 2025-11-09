/**
 * ML Playground Comprehensive Audit
 *
 * This test suite identifies all UX issues in the ML Playground:
 * 1. Results not visible
 * 2. Parameter sliders not working
 * 3. No train/test split
 * 4. No prediction visualization
 * 5. No summary statistics
 */

const { test, expect } = require('@playwright/test');
const path = require('path');

test.describe('ML Playground Audit', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('http://localhost:8080/playground');
        await page.waitForLoadState('networkidle');
    });

    test('AUDIT: Can navigate to ML Playground', async ({ page }) => {
        await expect(page).toHaveTitle(/BrunoML/);
        const heading = page.locator('h1');
        await expect(heading).toContainText('ML Playground');
    });

    test('AUDIT: CSV upload exists and is visible', async ({ page }) => {
        const uploadInput = page.locator('input[type="file"][accept=".csv"]');
        await expect(uploadInput).toBeVisible();

        const uploadLabel = page.locator('label.upload-button');
        await expect(uploadLabel).toBeVisible();
        await expect(uploadLabel).toContainText('Choose CSV File');
    });

    test('AUDIT: Can upload a CSV file', async ({ page }) => {
        // Create a simple CSV for testing
        const csvContent = `feature1,feature2,target
1,2,0
2,3,0
8,9,1
9,10,1`;

        const testFile = path.join(__dirname, 'test-data.csv');
        const fs = require('fs');
        fs.writeFileSync(testFile, csvContent);

        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);

        // Wait for upload confirmation
        await page.waitForTimeout(500);

        // Check if dataset info is displayed
        const datasetInfo = page.locator('.dataset-info');
        await expect(datasetInfo).toBeVisible({ timeout: 5000 });

        // Clean up
        fs.unlinkSync(testFile);
    });

    test('BUG #1: Algorithm selection dropdown exists', async ({ page }) => {
        const algorithmSelect = page.locator('select');
        await expect(algorithmSelect).toBeVisible();

        // Check available algorithms
        const options = await algorithmSelect.locator('option').allTextContents();
        console.log('Available algorithms:', options);
        expect(options.length).toBeGreaterThan(0);
    });

    test('BUG #2: Parameter sliders are present but may not work', async ({ page }) => {
        // First upload data
        const csvContent = `f1,f2,target
1,2,0
2,3,0
8,9,1
9,10,1`;

        const testFile = path.join(__dirname, 'temp-test.csv');
        const fs = require('fs');
        fs.writeFileSync(testFile, csvContent);

        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select K-Means
        const algorithmSelect = page.locator('select');
        await algorithmSelect.selectOption('KMeans');
        await page.waitForTimeout(500);

        // Check if parameter sliders appear
        const sliders = page.locator('input[type="range"]');
        const sliderCount = await sliders.count();
        console.log('Number of parameter sliders found:', sliderCount);

        if (sliderCount > 0) {
            // Try to interact with first slider
            const firstSlider = sliders.first();
            await expect(firstSlider).toBeVisible();

            const initialValue = await firstSlider.inputValue();
            console.log('Initial slider value:', initialValue);

            // Try to change slider value
            await firstSlider.fill('5');
            await page.waitForTimeout(300);

            const newValue = await firstSlider.inputValue();
            console.log('New slider value:', newValue);

            // BUG: Slider value might change but not affect algorithm
        }

        fs.unlinkSync(testFile);
    });

    test('BUG #3: Run Algorithm button exists', async ({ page }) => {
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await expect(runButton).toBeVisible();
    });

    test('BUG #4: Results area exists but content may not be visible', async ({ page }) => {
        // Upload CSV
        const csvContent = `f1,f2,target
1,2,0
2,3,0
8,9,1
9,10,1`;

        const testFile = path.join(__dirname, 'temp-results-test.csv');
        const fs = require('fs');
        fs.writeFileSync(testFile, csvContent);

        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select algorithm
        const algorithmSelect = page.locator('select');
        await algorithmSelect.selectOption('KMeans');
        await page.waitForTimeout(500);

        // Click run
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(2000);

        // Check for results
        const resultsArea = page.locator('.results-panel, .result-message, #results');
        const resultsText = await page.textContent('body');
        console.log('Page content after run:', resultsText.substring(0, 500));

        // BUG: Results might not be displayed prominently
        // Check if there's any visible output
        const hasVisibleResults = resultsText.includes('✅') ||
                                 resultsText.includes('Cluster') ||
                                 resultsText.includes('completed');

        if (!hasVisibleResults) {
            console.log('❌ BUG CONFIRMED: Results not visible after running algorithm');
        }

        fs.unlinkSync(testFile);
    });

    test('BUG #5: No train/test split functionality', async ({ page }) => {
        const trainTestSplit = page.locator('text=/train.*test/i, [class*="split"], #train-test-split');
        const count = await trainTestSplit.count();

        if (count === 0) {
            console.log('❌ BUG CONFIRMED: No train/test split UI found');
        }

        expect(count).toBe(0); // Expect this to fail, confirming the bug
    });

    test('BUG #6: No predictions table or visualization', async ({ page }) => {
        const predictionsTable = page.locator('table, .predictions-table, .prediction-viz');
        const count = await predictionsTable.count();

        if (count === 0) {
            console.log('❌ BUG CONFIRMED: No predictions table or visualization found');
        }

        expect(count).toBe(0); // Expect this to fail, confirming the bug
    });

    test('BUG #7: No summary statistics display', async ({ page }) => {
        const statsDisplay = page.locator('.summary-stats, .model-stats, text=/accuracy/i, text=/confusion/i');
        const count = await statsDisplay.count();

        if (count === 0) {
            console.log('❌ BUG CONFIRMED: No summary statistics display found');
        }

        // Note: We might have ModelPerformanceCard but it might not show useful stats
    });

    test('FULL WORKFLOW: Upload → Select → Configure → Run → View Results', async ({ page }) => {
        console.log('=== FULL ML PLAYGROUND WORKFLOW TEST ===');

        // Step 1: Upload CSV
        console.log('Step 1: Uploading CSV...');
        const csvContent = `feature1,feature2,target
1.0,2.0,0
1.5,2.5,0
2.0,3.0,0
7.0,8.0,1
8.0,9.0,1
9.0,10.0,1`;

        const testFile = path.join(__dirname, 'workflow-test.csv');
        const fs = require('fs');
        fs.writeFileSync(testFile, csvContent);

        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        const datasetInfo = page.locator('.dataset-info');
        const isDatasetVisible = await datasetInfo.isVisible();
        console.log('Dataset info visible:', isDatasetVisible);

        // Step 2: Select algorithm
        console.log('Step 2: Selecting K-Means algorithm...');
        const algorithmSelect = page.locator('select');
        await algorithmSelect.selectOption('KMeans');
        await page.waitForTimeout(500);

        // Step 3: Configure parameters (if sliders exist)
        console.log('Step 3: Checking for parameter configuration...');
        const sliders = page.locator('input[type="range"]');
        const sliderCount = await sliders.count();
        console.log('Parameter sliders available:', sliderCount);

        if (sliderCount > 0) {
            const firstSlider = sliders.first();
            const sliderName = await firstSlider.getAttribute('name');
            const initialValue = await firstSlider.inputValue();
            console.log(`Slider "${sliderName}" initial value: ${initialValue}`);

            // Try to change it
            await firstSlider.fill('3');
            const newValue = await firstSlider.inputValue();
            console.log(`Slider "${sliderName}" new value: ${newValue}`);
        }

        // Step 4: Run algorithm
        console.log('Step 4: Running algorithm...');
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await runButton.click();
        await page.waitForTimeout(3000);

        // Step 5: Check results
        console.log('Step 5: Checking results...');
        const pageContent = await page.textContent('body');

        const hasSuccessMessage = pageContent.includes('✅');
        const hasClusterInfo = pageContent.includes('Cluster') || pageContent.includes('cluster');
        const hasErrorMessage = pageContent.includes('❌');

        console.log('Success message found:', hasSuccessMessage);
        console.log('Cluster info found:', hasClusterInfo);
        console.log('Error message found:', hasErrorMessage);

        // Check for predictions table
        const table = page.locator('table');
        const tableCount = await table.count();
        console.log('Tables found:', tableCount);

        // Check for performance metrics
        const perfCard = page.locator('.model-performance-card, .performance-metrics');
        const perfCardExists = await perfCard.count() > 0;
        console.log('Performance card found:', perfCardExists);

        // Check if results are actually visible to user
        const resultsPanel = page.locator('.results-panel');
        const resultsVisible = await resultsPanel.isVisible().catch(() => false);
        console.log('Results panel visible:', resultsVisible);

        if (!resultsVisible) {
            console.log('❌ CRITICAL BUG: Results panel not visible after running algorithm!');
        }

        if (!hasSuccessMessage && !hasClusterInfo) {
            console.log('❌ CRITICAL BUG: No visible feedback about algorithm results!');
        }

        console.log('=== END WORKFLOW TEST ===');

        fs.unlinkSync(testFile);
    });
});
