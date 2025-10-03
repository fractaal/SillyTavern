/**
 * E2E tests for /delrange slash command
 * Requires the SillyTavern dev server running at ST_URL (set in jest.setup.js)
 */

describe('/delrange slash command', () => {
  beforeAll(async () => {
    await page.goto(global.ST_URL, { waitUntil: 'load' });
    // Wait for preloader to disappear
    await page.waitForSelector('#preloader', { hidden: true, timeout: 30000 });
  });

  const seedChat = async (count) => {
    await page.evaluate(async (n) => {
      const app = await import('/script.js');
      const $container = window.jQuery('#chat');
      $container.empty();
      app.chat.length = 0;
      for (let i = 0; i < n; i++) {
        app.chat.push({ mes: `Msg ${i}`, is_user: false, is_system: false, swipes: [] });
        const el = document.createElement('div');
        el.className = 'mes';
        el.setAttribute('mesid', String(i));
        const del = document.createElement('div');
        del.className = 'mes_edit_delete';
        el.appendChild(del);
        $container.append(el);
      }
    }, count);
  };

  const getChatMessages = async () => {
    return await page.evaluate(async () => {
      const app = await import('/script.js');
      return app.chat.map(m => m.mes);
    });
  };

  const runCommand = async (text) => {
    return await page.evaluate(async (cmd) => {
      const app = await import('/script.js');
      return await app.processCommands(cmd);
    }, text);
  };

  test('deletes a simple range (2-4) with force=true', async () => {
    await seedChat(7);

    await runCommand('/delrange 2-4 force=true');

    const msgs = await getChatMessages();
    expect(msgs).toEqual(['Msg 0', 'Msg 1', 'Msg 5', 'Msg 6']);

    // DOM reflects 4 remaining messages
    const domCount = await page.evaluate(() => document.querySelectorAll('#chat .mes').length);
    expect(domCount).toBe(4);
  });

  test('single index behaves like range of one (3)', async () => {
    await seedChat(5);
    await runCommand('/delrange 3 force=true');
    const msgs = await getChatMessages();
    expect(msgs).toEqual(['Msg 0', 'Msg 1', 'Msg 2', 'Msg 4']);
  });

  test('invalid reversed range is ignored (4-2)', async () => {
    await seedChat(6);
    await runCommand('/delrange 4-2 force=true');
    const msgs = await getChatMessages();
    expect(msgs).toEqual(['Msg 0', 'Msg 1', 'Msg 2', 'Msg 3', 'Msg 4', 'Msg 5']);
  });

  test('out-of-bounds range is ignored (10-12)', async () => {
    await seedChat(5);
    await runCommand('/delrange 10-12 force=true');
    const msgs = await getChatMessages();
    expect(msgs).toEqual(['Msg 0', 'Msg 1', 'Msg 2', 'Msg 3', 'Msg 4']);
  });

  test('confirmation appears when force not provided and can be cancelled', async () => {
    await seedChat(5);
    // Fire command without force
    await runCommand('/delrange 1-2');

    // Expect a dialog to exist; click Cancel
    // Confirm popup uses <dialog> with buttons; select the last open dialog and click a button that cancels
    await page.waitForSelector('dialog[open]:not([closing])', { timeout: 10000 });
    // Click Cancel if present, otherwise the button that is not affirmative
    const cancelled = await page.evaluate(async () => {
      const dlg = Array.from(document.querySelectorAll('dialog[open]:not([closing])')).pop();
      if (!dlg) return false;
      // Try to find a button marked as cancel
      const cancelBtn = dlg.querySelector('[data-result="0"], .cancel, button[aria-label="Cancel"], button:has(span:contains("Cancel"))');
      if (cancelBtn instanceof HTMLElement) {
        cancelBtn.click();
        return true;
      }
      // Fallback: click the last button
      const btn = Array.from(dlg.querySelectorAll('button')).pop();
      if (btn) { btn.click(); return true; }
      return false;
    });

    expect(cancelled).toBe(true);

    const msgs = await getChatMessages();
    expect(msgs).toEqual(['Msg 0', 'Msg 1', 'Msg 2', 'Msg 3', 'Msg 4']);
  });
});

