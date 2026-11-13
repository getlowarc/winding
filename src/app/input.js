// Input (engine.input): keyboard, mouse and gamepads, read by name each frame
// rather than handled as events -- what a game loop wants: is jump held, was
// it pressed since the last step, how far is the stick over.
//
// One namespace of names: keys are KeyboardEvent.code ('KeyW', 'Space',
// 'ShiftLeft'), so they stay where they are on any keyboard layout; mouse
// buttons are 'MouseLeft', 'MouseRight' and 'MouseMiddle'; a gamepad's are
// the standard mapping's ('PadA', 'PadLB', 'PadLeftX'), or 'PadButton7' and
// 'PadAxis2' for one with no standard mapping. Actions (bind) put several
// together: jump on Space or the A button, move on WASD or the left stick.

/** The standard gamepad mapping's buttons, by index (W3C Gamepad, "standard"). */
export const PAD_BUTTONS = ['PadA', 'PadB', 'PadX', 'PadY', 'PadLB', 'PadRB', 'PadLT', 'PadRT', 'PadBack', 'PadStart',
  'PadLS', 'PadRS', 'PadUp', 'PadDown', 'PadLeft', 'PadRight', 'PadHome'];
/** And its axes: the two sticks, x then y, y down. */
export const PAD_AXES = ['PadLeftX', 'PadLeftY', 'PadRightX', 'PadRightY'];
const MOUSE_BUTTONS = ['MouseLeft', 'MouseMiddle', 'MouseRight'];

export class Input {
  /**
   * @param keys     where key events come from: the window, so the canvas need not have focus
   * @param pointer  the element mouse positions are measured against: the canvas
   * @param pads     what lists the gamepads: navigator.getGamepads
   */
  constructor({ keys = globalThis, pointer = null, pads = () => globalThis.navigator?.getGamepads?.() ?? [], deadZone = 0.15 } = {}) {
    this._keys = keys;
    this._pointer = pointer;
    this._pads = pads;
    /** How far a stick must lean before it counts, 0 to 1: worn sticks rest a little off centre. */
    this.deadZone = deadZone;
    this._held = new Set();
    this._pressed = new Set();
    this._released = new Set();
    /** Per gamepad index: its buttons' values and its axes, as last polled. */
    this._padState = [];
    this._actions = new Map();
    /** The mouse over the pointer element, in CSS pixels from its top left. */
    this.mouse = { x: 0, y: 0, dx: 0, dy: 0, wheel: 0 };
    this._listeners = [];
    const on = (target, type, fn, options) => {
      if (!target?.addEventListener) return;
      target.addEventListener(type, fn, options);
      this._listeners.push([target, type, fn, options]);
    };
    on(keys, 'keydown', (e) => { if (!e.repeat) this._down(e.code); });
    on(keys, 'keyup', (e) => this._up(e.code));
    // Focus lost with a key held: its keyup goes to another window, and the key stays down for good.
    on(keys, 'blur', () => { for (const name of [...this._held]) this._up(name); });
    on(pointer, 'pointerdown', (e) => this._down(MOUSE_BUTTONS[e.button] ?? `Mouse${e.button}`));
    on(keys, 'pointerup', (e) => this._up(MOUSE_BUTTONS[e.button] ?? `Mouse${e.button}`));
    on(keys, 'pointermove', (e) => {
      const r = pointer?.getBoundingClientRect?.();
      this.mouse.x = e.clientX - (r?.left ?? 0);
      this.mouse.y = e.clientY - (r?.top ?? 0);
      this.mouse.dx += e.movementX ?? 0;
      this.mouse.dy += e.movementY ?? 0;
    });
    on(pointer, 'wheel', (e) => { this.mouse.wheel += e.deltaY; }, { passive: true });
  }

  _down(name) {
    if (this._held.has(name)) return;
    this._held.add(name);
    this._pressed.add(name);
  }

  _up(name) {
    if (!this._held.delete(name)) return;
    this._released.add(name);
  }

  /**
   * Start a frame, for a loop of your own: what was pressed and released
   * since the last one, and the gamepads polled. engine.run does this for
   * you, so that each press is seen by one fixed step: see consume().
   */
  update() {
    this.consume();
    this.poll();
  }

  /** The gamepads, read: their buttons' presses and releases, by index and in general. */
  poll() {
    const pads = this._pads() ?? [];
    for (let i = 0; i < Math.max(pads.length, this._padState.length); i++) {
      const pad = pads[i];
      const before = this._padState[i];
      if (!pad?.connected) {
        if (before) for (let b = 0; b < before.buttons.length; b++) if (before.buttons[b] > 0.5) this._padEdge(before, b, false, i);
        this._padState[i] = undefined;
        continue;
      }
      const state = before ?? { standard: pad.mapping === 'standard', buttons: [], axes: [] };
      for (let b = 0; b < pad.buttons.length; b++) {
        const value = pad.buttons[b].value ?? (pad.buttons[b].pressed ? 1 : 0);
        const was = (state.buttons[b] ?? 0) > 0.5, is = value > 0.5;
        state.buttons[b] = value;
        if (was !== is) this._padEdge(state, b, is, i);
      }
      state.axes = Array.from(pad.axes);
      this._padState[i] = state;
    }
  }

  _padEdge(state, b, down, i) {
    for (const name of [state.standard ? PAD_BUTTONS[b] : undefined, `PadButton${b}`]) {
      if (name === undefined) continue;
      (down ? this._pressed : this._released).add(name).add(`${name}#${i}`);
    }
  }

  /** Presses, releases and mouse movement handed out: a fixed step has seen them. */
  consume() {
    this._pressed.clear();
    this._released.clear();
    this.mouse.dx = 0;
    this.mouse.dy = 0;
    this.mouse.wheel = 0;
  }

  /**
   * An action: a name for several inputs together. Buttons: `bind('jump',
   * ['Space', 'PadA'])`. An axis from -1 to 1: `bind('moveX', { negative:
   * ['KeyA'], positive: ['KeyD'], axis: 'PadLeftX' })`.
   */
  bind(action, sources) {
    if (Array.isArray(sources)) sources = { buttons: sources };
    if (!(sources && typeof sources === 'object')) throw new Error(`bind: ${action} needs a list of names, or { negative, positive, axis }`);
    this._actions.set(action, { buttons: sources.buttons ?? [], negative: sources.negative ?? [], positive: sources.positive ?? [], axis: sources.axis ?? null });
    return this;
  }

  /** Whether a key, button or action is held. `pad`: one gamepad's, by index; else any. */
  down(name, pad) {
    const action = this._actions.get(name);
    if (action) return action.buttons.some((n) => this.down(n, pad)) || Math.abs(this._axisAction(action, pad)) > 0.5;
    return this.value(name, pad) > 0.5;
  }

  /** Whether it went down since the last fixed step (or update()). */
  pressed(name, pad) {
    const action = this._actions.get(name);
    if (action) return [...action.buttons, ...action.negative, ...action.positive].some((n) => this.pressed(n, pad));
    return this._pressed.has(pad === undefined ? name : `${name}#${pad}`);
  }

  /** Whether it came up since the last fixed step (or update()). */
  released(name, pad) {
    const action = this._actions.get(name);
    if (action) return [...action.buttons, ...action.negative, ...action.positive].some((n) => this.released(n, pad));
    return this._released.has(pad === undefined ? name : `${name}#${pad}`);
  }

  /**
   * How far: 0 or 1 for a key, 0 to 1 for a trigger, -1 to 1 for a stick's
   * axis or an axis action. A stick inside the dead zone reads 0, and past it
   * starts again from 0, so there is no jump at its edge.
   */
  value(name, pad) {
    const action = this._actions.get(name);
    if (action) {
      if (action.negative.length || action.positive.length || action.axis) return this._axisAction(action, pad);
      return Math.max(0, ...action.buttons.map((n) => this.value(n, pad)));
    }
    if (!name.startsWith('Pad')) return pad === undefined && this._held.has(name) ? 1 : 0;
    let best = 0;
    for (let i = 0; i < this._padState.length; i++) {
      if (pad !== undefined && i !== pad) continue;
      const v = this._padValue(this._padState[i], name);
      if (Math.abs(v) > Math.abs(best)) best = v;
    }
    return best;
  }

  _padValue(state, name) {
    if (!state) return 0;
    let b = state.standard ? PAD_BUTTONS.indexOf(name) : -1;
    if (b < 0 && name.startsWith('PadButton')) b = Number(name.slice(9));
    if (b >= 0) return state.buttons[b] ?? 0;
    let a = state.standard ? PAD_AXES.indexOf(name) : -1;
    if (a < 0 && name.startsWith('PadAxis')) a = Number(name.slice(7));
    if (!(a >= 0)) return 0;
    // Radial over the stick the axis belongs to: a stick pushed along a
    // diagonal is not cut short on one axis by the other's dead zone.
    const pair = a - (a % 2), x = state.axes[pair] ?? 0, y = state.axes[pair + 1] ?? 0;
    const length = Math.hypot(x, y);
    if (length <= this.deadZone) return 0;
    const scale = Math.min(1, (length - this.deadZone) / (1 - this.deadZone)) / length;
    return (state.axes[a] ?? 0) * scale;
  }

  _axisAction(action, pad) {
    const sum = action.positive.reduce((s, n) => s + this.value(n, pad), 0) - action.negative.reduce((s, n) => s + this.value(n, pad), 0)
      + (action.axis ? this.value(action.axis, pad) : 0);
    return Math.max(-1, Math.min(1, sum));
  }

  /** The gamepads connected, by index: { index, id, standard }. */
  get pads() {
    const out = [];
    for (const pad of this._pads() ?? []) if (pad?.connected) out.push({ index: pad.index, id: pad.id, standard: pad.mapping === 'standard' });
    return out;
  }

  /** Capture the mouse for a camera that turns with it: mouse.dx and dy keep counting past the screen's edge. Call from a click. */
  lockPointer() {
    return this._pointer?.requestPointerLock?.();
  }

  /** Whether the mouse is captured. */
  get locked() {
    return this._pointer !== null && globalThis.document?.pointerLockElement === this._pointer;
  }

  /** Stop listening. */
  destroy() {
    for (const [target, type, fn, options] of this._listeners) target.removeEventListener(type, fn, options);
    this._listeners.length = 0;
  }
}
