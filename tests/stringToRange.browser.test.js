/**
 * Browser unit tests for stringToRange via module import in the app context.
 */

describe('stringToRange parsing (browser module)', () => {
  beforeAll(async () => {
    await page.goto(global.ST_URL, { waitUntil: 'load' });
    await page.waitForSelector('#preloader', { hidden: true, timeout: 30000 });
  });

  const parse = async (input, min, max) => {
    return await page.evaluate(async (inp, lo, hi) => {
      const mod = await import('/scripts/utils.js');
      const result = mod.stringToRange(inp, lo, hi);
      if (result === null) return null;
      return { start: result.start, end: result.end };
    }, input, min, max);
  };

  test('single number within bounds', async () => {
    await expect(parse('5', 0, 10)).resolves.toEqual({ start: 5, end: 5 });
  });

  test('range within bounds', async () => {
    await expect(parse('2-7', 0, 10)).resolves.toEqual({ start: 2, end: 7 });
  });

  test('invalid reversed range', async () => {
    await expect(parse('7-2', 0, 10)).resolves.toBeNull();
  });

  test('non-numeric input returns null', async () => {
    await expect(parse('abc', 0, 10)).resolves.toBeNull();
  });

  test('out-of-bounds returns null', async () => {
    await expect(parse('0-11', 0, 10)).resolves.toBeNull();
    await expect(parse('-1', 0, 10)).resolves.toBeNull();
  });
});

