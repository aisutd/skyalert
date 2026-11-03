// Tiny synthesized sound effects for the map (off by default): no audio files needed

const SOUND_KEY = 'skyalert-sound'
let audio = null

// Wrapped in try/catch because some browsers (private mode, blocked storage) throw
export function loadSoundSetting() {
  try {
    return localStorage.getItem(SOUND_KEY) === 'on'
  } catch {
    return false
  }
}

export function saveSoundSetting(on) {
  try {
    localStorage.setItem(SOUND_KEY, on ? 'on' : 'off')
  } catch {
    // not remembered next time
  }
}

// kind: 'open' | 'switch' | 'close'
export function playSound(kind) {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)()
    if (audio.state === 'suspended') audio.resume()
    const now = audio.currentTime
    const out = audio.createGain()
    out.gain.value = 0.045
    out.connect(audio.destination)
    const tone = (from, to, start, length, type = 'sine') => {
      const osc = audio.createOscillator()
      const env = audio.createGain()
      osc.type = type
      osc.frequency.setValueAtTime(from, now + start)
      osc.frequency.exponentialRampToValueAtTime(to, now + start + length)
      env.gain.setValueAtTime(0.0001, now + start)
      env.gain.exponentialRampToValueAtTime(1, now + start + 0.015)
      env.gain.exponentialRampToValueAtTime(0.0001, now + start + length)
      osc.connect(env).connect(out)
      osc.start(now + start)
      osc.stop(now + start + length + 0.02)
    }
    if (kind === 'open') { tone(520, 880, 0, 0.32); tone(1320, 1760, 0.12, 0.38) }
    else if (kind === 'switch') tone(880, 1180, 0, 0.18, 'triangle')
    else if (kind === 'close') tone(760, 420, 0, 0.26)
  } catch {
    // audio not available; stay silent
  }
}
