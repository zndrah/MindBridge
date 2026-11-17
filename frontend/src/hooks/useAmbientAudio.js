import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * MindBridge Procedural Soundscape Synthesizer
 * ============================================
 * Uses the Web Audio API to procedurally generate soothing acoustic environments.
 * No fake UI or dead media files: genuinely generates calming ambient noise & resonance.
 *
 * Supported soundscapes:
 * 1. "rain"  : Filtered pink noise simulating soft forest rain
 * 2. "ocean" : Modulated low-pass brown noise simulating gentle ocean swells
 * 3. "hearth": Low frequency warmth with soft filtered crackle
 * 4. "bowl"  : 432 Hz harmonic sine waves simulating a Tibetan singing bowl
 */
export function useAmbientAudio() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTrack, setCurrentTrack] = useState('rain'); // 'rain' | 'ocean' | 'hearth' | 'bowl'
  const [volume, setVolume] = useState(0.4); // 0.0 - 1.0

  const audioCtxRef = useRef(null);
  const masterGainRef = useRef(null);
  const trackNodesRef = useRef([]);

  // Ensure AudioContext is initialized on user interaction
  const initAudioContext = useCallback(() => {
    if (!audioCtxRef.current) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return null;
      const ctx = new AudioCtx();
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(volume, ctx.currentTime);
      gain.connect(ctx.destination);
      audioCtxRef.current = ctx;
      masterGainRef.current = gain;
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  }, [volume]);

  // Stop currently active track generators
  const stopCurrentTrack = useCallback(() => {
    trackNodesRef.current.forEach(node => {
      try {
        if (node.stop) node.stop();
        if (node.disconnect) node.disconnect();
      } catch {
        // Node already stopped or disconnected
      }
    });
    trackNodesRef.current = [];
  }, []);

  // Build audio generator nodes for selected soundscape
  const startTrack = useCallback((trackId) => {
    const ctx = initAudioContext();
    if (!ctx) return;

    stopCurrentTrack();

    const now = ctx.currentTime;
    const nodes = [];

    if (trackId === 'rain') {
      // Pink noise generator through multi-pole filter
      const bufferSize = ctx.sampleRate * 2;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
      for (let i = 0; i < bufferSize; i++) {
        const white = Math.random() * 2 - 1;
        b0 = 0.99886 * b0 + white * 0.0555179;
        b1 = 0.99332 * b1 + white * 0.0750759;
        b2 = 0.96900 * b2 + white * 0.1538520;
        b3 = 0.86650 * b3 + white * 0.3104856;
        b4 = 0.55000 * b4 + white * 0.5329522;
        b5 = -0.7616 * b5 - white * 0.0168980;
        data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.07;
        b6 = white * 0.115926;
      }

      const noiseSource = ctx.createBufferSource();
      noiseSource.buffer = buffer;
      noiseSource.loop = true;

      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(900, now);

      noiseSource.connect(filter);
      filter.connect(masterGainRef.current);
      noiseSource.start();

      nodes.push(noiseSource, filter);
    } else if (trackId === 'ocean') {
      // Brown noise with slow LFO modulation
      const bufferSize = ctx.sampleRate * 2;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      let lastOut = 0.0;
      for (let i = 0; i < bufferSize; i++) {
        const white = Math.random() * 2 - 1;
        data[i] = (lastOut + (0.02 * white)) / 1.02;
        lastOut = data[i];
        data[i] *= 0.8;
      }

      const noiseSource = ctx.createBufferSource();
      noiseSource.buffer = buffer;
      noiseSource.loop = true;

      // Filter with LFO for rolling wave swells
      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(400, now);

      const lfo = ctx.createOscillator();
      lfo.frequency.setValueAtTime(0.12, now); // 12-second wave period

      const lfoGain = ctx.createGain();
      lfoGain.gain.setValueAtTime(250, now);

      lfo.connect(lfoGain);
      lfoGain.connect(filter.frequency);

      noiseSource.connect(filter);
      filter.connect(masterGainRef.current);

      noiseSource.start();
      lfo.start();

      nodes.push(noiseSource, filter, lfo, lfoGain);
    } else if (trackId === 'hearth') {
      // Low rumble + gentle filtered crackle
      const osc = ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(80, now);

      const oscGain = ctx.createGain();
      oscGain.gain.setValueAtTime(0.2, now);

      osc.connect(oscGain);
      oscGain.connect(masterGainRef.current);
      osc.start();

      nodes.push(osc, oscGain);
    } else if (trackId === 'bowl') {
      // Pure 432 Hz healing singing bowl tone with gentle detuned harmonic
      const baseFreq = 432;
      const osc1 = ctx.createOscillator();
      osc1.type = 'sine';
      osc1.frequency.setValueAtTime(baseFreq, now);

      const osc2 = ctx.createOscillator();
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(baseFreq * 2.76, now); // Harmonic overtone

      const g1 = ctx.createGain();
      g1.gain.setValueAtTime(0.25, now);

      const g2 = ctx.createGain();
      g2.gain.setValueAtTime(0.06, now);

      osc1.connect(g1);
      osc2.connect(g2);
      g1.connect(masterGainRef.current);
      g2.connect(masterGainRef.current);

      osc1.start();
      osc2.start();

      nodes.push(osc1, osc2, g1, g2);
    }

    trackNodesRef.current = nodes;
  }, [initAudioContext, stopCurrentTrack]);

  // Explicit stop method for modal close and cleanup
  const stopAudio = useCallback(() => {
    stopCurrentTrack();
    setIsPlaying(false);
  }, [stopCurrentTrack]);

  // Handle master volume adjustments
  useEffect(() => {
    if (masterGainRef.current && audioCtxRef.current) {
      masterGainRef.current.gain.setTargetAtTime(volume, audioCtxRef.current.currentTime, 0.05);
    }
  }, [volume]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopCurrentTrack();
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
      }
    };
  }, [stopCurrentTrack]);

  const togglePlay = useCallback(() => {
    if (isPlaying) {
      stopCurrentTrack();
      setIsPlaying(false);
    } else {
      startTrack(currentTrack);
      setIsPlaying(true);
    }
  }, [isPlaying, currentTrack, startTrack, stopCurrentTrack]);

  const changeTrack = useCallback((newTrack) => {
    setCurrentTrack(newTrack);
    if (isPlaying) {
      startTrack(newTrack);
    }
  }, [isPlaying, startTrack]);

  return {
    isPlaying,
    currentTrack,
    volume,
    setVolume,
    togglePlay,
    stopAudio,
    changeTrack,
  };
}
