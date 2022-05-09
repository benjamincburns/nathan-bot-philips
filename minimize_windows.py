import win32gui

def toggle_rl_windows(minimise=True):
    window_ledger = {}
    def winEnumHandler( hwnd, ctx ):
        if win32gui.IsWindowVisible(hwnd):
            if win32gui.GetWindowText(hwnd).find("Rocket League") != -1:
                window_ledger[hwnd] = win32gui.GetWindowText(hwnd)


    win32gui.EnumWindows(winEnumHandler, None)
    # show_codes: 9 to show, 6 to hide
    code = 6
    if not minimise:
        code = 9
    for k in window_ledger.keys():
        win32gui.ShowWindow(k, code)
        print(k)

toggle_rl_windows()
