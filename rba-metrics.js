(() => {
    const start = performance.now();

    // ---- Counters ----
    let focusChanges = 0,
        blurEvents = 0;
    let clickCount = 0,
        keyCount = 0;
    let totalKeyDelay = 0,
        lastKeyTime = null;
    let pointerDistance = 0,
        pointerEventCount = 0;
    let scrollDistance = 0,
        scrollEventCount = 0;
    let lastScroll = window.scrollY || document.documentElement.scrollTop;
    let lastMoveTime = 0,
        lastX = null,
        lastY = null;
    let firstKeyTime = null,
        firstClickTime = null;
    let pasteCount = 0,
        inputFocusCount = 0,
        resizeCount = 0;

    // ---- Idle tracking ----
    let lastActivity = performance.now();
    let idleTimeTotal = 0;
    let wasIdle = false;
    let idleStartTime = null;

    function recordActivity() {
        const now = performance.now();
        if (wasIdle && idleStartTime !== null) {
            // We were idle, now we're active - add the idle period
            idleTimeTotal += now - idleStartTime;
            wasIdle = false;
            idleStartTime = null;
        }
        lastActivity = now;
    }

    // Check for idle periods periodically
    setInterval(() => {
        const now = performance.now();
        const timeSinceActivity = now - lastActivity;

        // If it's been more than 100ms since last activity, consider it idle
        if (timeSinceActivity > 100 && !wasIdle) {
            wasIdle = true;
            idleStartTime = lastActivity;
        }
    }, 50);

    [
        "mousemove",
        "pointermove",
        "touchmove",
        "keydown",
        "click",
        "scroll",
        "focus",
    ].forEach((ev) =>
        document.addEventListener(ev, recordActivity, { passive: true }),
    );

    // ---- Event listeners ----
    document.addEventListener("visibilitychange", () => focusChanges++);
    window.addEventListener("focus", () => focusChanges++);
    window.addEventListener("blur", () => blurEvents++);
    document.addEventListener("click", () => {
        clickCount++;
        if (!firstClickTime) firstClickTime = performance.now();
    });

    document.addEventListener("keydown", () => {
        const now = performance.now();
        if (!firstKeyTime) firstKeyTime = now;
        if (lastKeyTime) totalKeyDelay += now - lastKeyTime;
        lastKeyTime = now;
        keyCount++;
    });

    document.addEventListener("paste", () => pasteCount++);
    document.addEventListener("focusin", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")
            inputFocusCount++;
    });

    const trackPointer = (e) => {
        const now = performance.now();
        if (now - lastMoveTime < 50) return; // throttle ~20 Hz
        pointerEventCount++;
        if (lastX !== null && lastY !== null) {
            pointerDistance += Math.hypot(e.pageX - lastX, e.pageY - lastY);
        }
        lastX = e.pageX;
        lastY = e.pageY;
        lastMoveTime = now;
    };
    document.addEventListener("mousemove", trackPointer, { passive: true });
    document.addEventListener("pointermove", trackPointer, { passive: true });
    document.addEventListener(
        "touchmove",
        (e) => {
            pointerEventCount++;
            const touch = e.touches && e.touches[0];
            if (!touch) return;
            const { pageX, pageY } = touch;
            if (lastX !== null && lastY !== null) {
                pointerDistance += Math.hypot(pageX - lastX, pageY - lastY);
            }
            lastX = pageX;
            lastY = pageY;
        },
        { passive: true },
    );

    window.addEventListener(
        "scroll",
        () => {
            const currentScroll =
                window.scrollY || document.documentElement.scrollTop;
            scrollDistance += Math.abs(currentScroll - lastScroll);
            lastScroll = currentScroll;
            scrollEventCount++;
        },
        { passive: true },
    );

    window.addEventListener("resize", () => resizeCount++);

    // ---- Device UUID management ----
    function generateUUIDv4() {
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
            const r = crypto.getRandomValues(new Uint8Array(1))[0] & 15;
            const v = c === "x" ? r : (r & 0x3) | 0x8;
            return v.toString(16);
        });
    }

    let deviceUUID = localStorage.getItem("rbaDeviceUUID");
    if (!deviceUUID) {
        deviceUUID = generateUUIDv4();
        localStorage.setItem("rbaDeviceUUID", deviceUUID);
    }

    // ---- On form submit, attach metrics ----
    const form = document.querySelector("form");
    const metricsField = document.getElementById("rbaMetricsField");

    if (form && metricsField) {
        form.addEventListener("submit", () => {
            const now = performance.now();
            const elapsed = Math.max(0, now - start);
            const avgKeyDelay = keyCount ? totalKeyDelay / keyCount : 0;

            // Account for final idle period if currently idle
            if (wasIdle && idleStartTime !== null) {
                idleTimeTotal += now - idleStartTime;
            }

            // --- Environment snapshot ---
            const env = {
                tz_offset_min: new Date().getTimezoneOffset(),
                language: navigator.language || "unknown",
                platform:
                    navigator.userAgentData?.platform || navigator.platform || "unknown",
                device_memory_gb: navigator.deviceMemory || null,
                hardware_concurrency: navigator.hardwareConcurrency || null,
                screen_width_px: screen.width,
                screen_height_px: screen.height,
                pixel_ratio: window.devicePixelRatio || 1,
                color_depth: screen.colorDepth || null,
                touch_support: "ontouchstart" in window,
                webauthn_supported: !!window.PublicKeyCredential,
            };

            // --- Metrics ---
            const metrics = {
                device_uuid: deviceUUID,
                focus_changes: focusChanges,
                blur_events: blurEvents,
                click_count: clickCount,
                key_count: keyCount,
                avg_key_delay_ms: Math.round(avgKeyDelay),
                pointer_distance_px: Math.round(pointerDistance),
                pointer_event_count: pointerEventCount,
                scroll_distance_px: Math.round(scrollDistance),
                scroll_event_count: scrollEventCount,
                time_to_first_key_ms: firstKeyTime
                    ? Math.round(firstKeyTime - start)
                    : null,
                time_to_first_click_ms: firstClickTime
                    ? Math.round(firstClickTime - start)
                    : null,
                total_session_time_ms: Math.round(elapsed),
                idle_time_total_ms: Math.round(idleTimeTotal),
                active_time_ms: Math.round(elapsed - idleTimeTotal),
                input_focus_count: inputFocusCount,
                paste_events: pasteCount,
                resize_events: resizeCount,
                metrics_version: 1,
                collection_timestamp: new Date().toISOString(),
                ...env,
            };

            try {
                metricsField.value = JSON.stringify(metrics);
            } catch (err) {
                console.error("RBA metrics serialization failed", err);
            }
        });
    } else {
        console.warn("RBA metrics script: form or #rbaMetricsField not found");
    }
})();
