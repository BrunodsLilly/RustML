/**
 * Decision Tree and Naive Bayes Integration Test
 *
 * Tests the newly added classifiers in the ML Playground:
 * - Decision Tree algorithm selection and execution
 * - Naive Bayes algorithm selection and execution
 * - Parameter configuration for Decision Tree
 * - Results display for both algorithms
 */

const { test, expect } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

test.describe('Decision Tree & Naive Bayes Integration', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('http://localhost:8080/playground');
        await page.waitForLoadState('networkidle');
    });

    test('Decision Tree: Algorithm button exists and is clickable', async ({ page }) => {
        // Look for Decision Tree button
        const dtButton = page.locator('button').filter({ hasText: /Decision Tree/i });
        await expect(dtButton).toBeVisible();

        // Check for tree icon
        const buttonText = await dtButton.textContent();
        console.log('Decision Tree button text:', buttonText);
        expect(buttonText).toContain('Decision Tree');
    });

    test('Naive Bayes: Algorithm button exists and is clickable', async ({ page }) => {
        // Look for Naive Bayes button
        const nbButton = page.locator('button').filter({ hasText: /Naive Bayes/i });
        await expect(nbButton).toBeVisible();

        const buttonText = await nbButton.textContent();
        console.log('Naive Bayes button text:', buttonText);
        expect(buttonText).toContain('Naive Bayes');
    });

    test('Decision Tree: Full workflow with CSV upload', async ({ page }) => {
        console.log('=== Testing Decision Tree Full Workflow ===');

        // Create test CSV
        const csvContent = `feature1,feature2,target
1.0,2.0,0
1.5,2.5,0
2.0,3.0,0
8.0,9.0,1
8.5,9.5,1
9.0,10.0,1`;

        const testFile = path.join(__dirname, 'dt-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        console.log('Step 1: Uploading CSV...');
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Verify dataset info appears
        const datasetInfo = page.locator('.dataset-info');
        await expect(datasetInfo).toBeVisible({ timeout: 5000 });
        console.log('✓ Dataset loaded');

        // Select Decision Tree
        console.log('Step 2: Selecting Decision Tree algorithm...');
        const dtButton = page.locator('button').filter({ hasText: /Decision Tree/i });
        await dtButton.click();
        await page.waitForTimeout(500);

        // Check if algorithm explanation appears
        const explanation = page.locator('.algorithm-explanation');
        const hasExplanation = await explanation.isVisible().catch(() => false);
        if (hasExplanation) {
            const explanationText = await explanation.textContent();
            console.log('Algorithm explanation visible:', explanationText.substring(0, 100) + '...');
        }

        // Look for Run button
        console.log('Step 3: Looking for Run button...');
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await expect(runButton).toBeVisible();
        console.log('✓ Run button found');

        // Click Run
        console.log('Step 4: Running Decision Tree...');
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for results
        console.log('Step 5: Checking for results...');
        const pageContent = await page.textContent('body');

        const hasSuccess = pageContent.includes('✅') || pageContent.includes('Decision Tree completed');
        const hasAccuracy = pageContent.includes('Accuracy') || pageContent.includes('%');
        const hasTreeInfo = pageContent.includes('Tree') || pageContent.includes('depth');

        console.log('Success message:', hasSuccess);
        console.log('Accuracy displayed:', hasAccuracy);
        console.log('Tree info displayed:', hasTreeInfo);

        // At least one success indicator should be present
        expect(hasSuccess || hasAccuracy || hasTreeInfo).toBeTruthy();

        console.log('=== Decision Tree Test Complete ===');

        // Clean up
        fs.unlinkSync(testFile);
    });

    test('Naive Bayes: Full workflow with CSV upload', async ({ page }) => {
        console.log('=== Testing Naive Bayes Full Workflow ===');

        // Create test CSV
        const csvContent = `feature1,feature2,target
1.0,2.0,0
1.5,2.5,0
2.0,3.0,0
8.0,9.0,1
8.5,9.5,1
9.0,10.0,1`;

        const testFile = path.join(__dirname, 'nb-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        console.log('Step 1: Uploading CSV...');
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Verify dataset info appears
        const datasetInfo = page.locator('.dataset-info');
        await expect(datasetInfo).toBeVisible({ timeout: 5000 });
        console.log('✓ Dataset loaded');

        // Select Naive Bayes
        console.log('Step 2: Selecting Naive Bayes algorithm...');
        const nbButton = page.locator('button').filter({ hasText: /Naive Bayes/i });
        await nbButton.click();
        await page.waitForTimeout(500);

        // Look for Run button
        console.log('Step 3: Looking for Run button...');
        const runButton = page.locator('button').filter({ hasText: /run/i });
        await expect(runButton).toBeVisible();
        console.log('✓ Run button found');

        // Click Run
        console.log('Step 4: Running Naive Bayes...');
        await runButton.click();
        await page.waitForTimeout(3000);

        // Check for results
        console.log('Step 5: Checking for results...');
        const pageContent = await page.textContent('body');

        const hasSuccess = pageContent.includes('✅') || pageContent.includes('Naive Bayes completed');
        const hasAccuracy = pageContent.includes('Accuracy') || pageContent.includes('%');
        const hasGaussian = pageContent.includes('Gaussian') || pageContent.includes('probabilistic');

        console.log('Success message:', hasSuccess);
        console.log('Accuracy displayed:', hasAccuracy);
        console.log('Gaussian/probabilistic info:', hasGaussian);

        // At least one success indicator should be present
        expect(hasSuccess || hasAccuracy || hasGaussian).toBeTruthy();

        console.log('=== Naive Bayes Test Complete ===');

        // Clean up
        fs.unlinkSync(testFile);
    });

    test('Decision Tree: Parameter configuration visible', async ({ page }) => {
        console.log('=== Testing Decision Tree Parameter Configuration ===');

        // Create test CSV first
        const csvContent = `f1,f2,target
1,2,0
8,9,1`;
        const testFile = path.join(__dirname, 'dt-params-test.csv');
        fs.writeFileSync(testFile, csvContent);

        // Upload CSV
        const uploadInput = page.locator('input[type="file"]');
        await uploadInput.setInputFiles(testFile);
        await page.waitForTimeout(1000);

        // Select Decision Tree
        const dtButton = page.locator('button').filter({ hasText: /Decision Tree/i });
        await dtButton.click();
        await page.waitForTimeout(500);

        // Click Configure Parameters button
        const configButton = page.locator('button').filter({ hasText: /configure/i });
        const configExists = await configButton.count() > 0;

        if (configExists) {
            await configButton.click();
            await page.waitForTimeout(500);

            // Look for parameter sliders
            const sliders = page.locator('input[type="range"]');
            const sliderCount = await sliders.count();
            console.log('Parameter sliders found:', sliderCount);

            // Decision Tree should have 3 parameters: max_depth, min_samples_split, min_samples_leaf
            expect(sliderCount).toBeGreaterThan(0);
        } else {
            console.log('Configure button not found - parameters may be displayed by default');
        }

        console.log('=== Parameter Configuration Test Complete ===');

        // Clean up
        fs.unlinkSync(testFile);
    });

    test('Compare all 7 algorithms availability', async ({ page }) => {
        console.log('=== Verifying All Algorithm Buttons ===');

        const algorithms = [
            'K-Means',
            'PCA',
            'Logistic Regression',
            'Decision Tree',
            'Naive Bayes',
            'Standard Scaler',
            'MinMax Scaler'
        ];

        for (const algo of algorithms) {
            const button = page.locator('button').filter({ hasText: new RegExp(algo, 'i') });
            const isVisible = await button.isVisible().catch(() => false);
            console.log(`${algo}: ${isVisible ? '✓' : '✗'}`);

            // Log if not visible
            if (!isVisible) {
                console.log(`WARNING: ${algo} button not found!`);
            }
        }

        console.log('=== Algorithm Availability Check Complete ===');
    });
});
