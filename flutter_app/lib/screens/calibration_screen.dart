import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';
import '../theme/app_theme.dart';
import '../utils/log.dart';
import '../widgets/app_banner.dart';
import '../widgets/connection_dot.dart';
import 'calib_images_screen.dart';

class CalibrationScreen extends StatefulWidget {
  const CalibrationScreen({super.key});

  @override
  State<CalibrationScreen> createState() => _CalibrationScreenState();
}

class _CalibrationScreenState extends State<CalibrationScreen> {
  static const _log = Log('CalibrationScreen');

  final _picker = ImagePicker();
  late final AppState _appState;
  bool _capturing = false;
  Timer? _countdownTimer;
  int _remainingMs = 0;

  CameraController? _camController;
  bool _camReady = false;

  @override
  void initState() {
    super.initState();
    _appState = context.read<AppState>();
    _appState.addListener(_onStateChange);
    _checkExistingTrigger();
  }

  @override
  void dispose() {
    _appState.removeListener(_onStateChange);
    _countdownTimer?.cancel();
    _camController?.dispose();
    super.dispose();
  }

  void _onStateChange() {
    if (_appState.isCamera && _appState.calibTriggerAt != null) {
      _startCountdown();
    }
  }

  void _checkExistingTrigger() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && _appState.isCamera && _appState.calibTriggerAt != null) {
        _startCountdown();
      }
    });
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty || !mounted) return;
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final ctrl = CameraController(back, ResolutionPreset.max, enableAudio: false);
      await ctrl.initialize();
      if (!mounted) {
        await ctrl.dispose();
        return;
      }
      setState(() {
        _camController = ctrl;
        _camReady = true;
      });
    } catch (e, st) {
      _log.warn('Nie udało się zainicjować kamery', e, st);
    }
  }

  Future<void> _disposeCamera() async {
    final ctrl = _camController;
    _camController = null;
    _camReady = false;
    await ctrl?.dispose();
  }

  void _startCountdown() {
    _countdownTimer?.cancel();
    final triggerAt = _appState.calibTriggerAt!;
    final offset = _appState.serverTimeOffset;
    _appState.clearCalibTrigger();

    int remainingMs() {
      final now = DateTime.now().millisecondsSinceEpoch / 1000.0;
      return ((triggerAt - now - offset) * 1000).round();
    }

    if (remainingMs() <= 0) {
      _log.warn('Pominięto nieaktualny trigger kalibracyjny '
          '(at=$triggerAt, offset=$offset, remaining=${remainingMs()}ms)');
      setState(() => _remainingMs = 0);
      return;
    }

    setState(() => _remainingMs = remainingMs());
    if (!kIsWeb) _initCamera(); // initialize camera in background during countdown

    _countdownTimer = Timer.periodic(const Duration(milliseconds: 50), (t) {
      if (!mounted) {
        t.cancel();
        return;
      }
      final remaining = remainingMs();
      if (remaining <= 0) {
        t.cancel();
        setState(() => _remainingMs = 0);
        _autoCapture();
      } else {
        setState(() => _remainingMs = remaining);
      }
    });
  }

  /// Takes photo automatically using CameraController (no user interaction).
  /// Falls back to image_picker if camera init failed (e.g., permission denied).
  Future<void> _autoCapture() async {
    if (_capturing) return;
    setState(() => _capturing = true);
    try {
      final XFile xfile;
      if (!kIsWeb && _camReady && _camController != null) {
        xfile = await _camController!.takePicture();
        _log.info('Automatyczne zdjęcie kalibracyjne wykonane');
      } else {
        _log.warn('Camera nie gotowa — fallback na image_picker');
        final picked = await _picker.pickImage(
          source: kIsWeb ? ImageSource.gallery : ImageSource.camera,
          imageQuality: 90,
        );
        if (picked == null) {
          _log.info('Zdjęcie kalibracyjne anulowane');
          return;
        }
        xfile = picked;
      }
      final bytes = await xfile.readAsBytes();
      final ok = await _appState.uploadCalibImageNow(bytes);
      if (mounted && ok) {
        final total = _appState.myDevice?.calibFrameCount ?? 0;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Klatka kalibracyjna przesłana — łącznie $total')),
        );
      }
    } catch (e, st) {
      _log.warn('Błąd automatycznego przechwytywania kalibracyjnego', e, st);
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Błąd aparatu: $e')));
      }
    } finally {
      await _disposeCamera();
      if (mounted) setState(() => _capturing = false);
    }
  }

  /// Manual capture via image_picker — user presses shutter themselves.
  Future<void> _manualCapture() async {
    if (_capturing) return;
    setState(() => _capturing = true);
    try {
      final xfile = await _picker.pickImage(
        source: kIsWeb ? ImageSource.gallery : ImageSource.camera,
        imageQuality: 90,
      );
      if (xfile == null) {
        _log.info('Ręczne zdjęcie kalibracyjne anulowane');
        return;
      }
      final bytes = await xfile.readAsBytes();
      final ok = await _appState.uploadCalibImageNow(bytes);
      if (mounted && ok) {
        final total = _appState.myDevice?.calibFrameCount ?? 0;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Klatka kalibracyjna przesłana — łącznie $total')),
        );
      }
    } catch (e, st) {
      _log.warn('Błąd ręcznego przechwytywania kalibracyjnego', e, st);
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Błąd aparatu: $e')));
      }
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  Future<void> _startCalibration(AppState state) async {
    await state.startCalibration();
    if (mounted && state.error == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Kalibracja uruchomiona — czekaj na wynik')),
      );
    }
  }

  Widget _buildCameraView(TextTheme tt) {
    return Stack(
      fit: StackFit.expand,
      children: [
        if (!kIsWeb && _camReady && _camController != null)
          CameraPreview(_camController!)
        else
          const ColoredBox(color: Colors.black),
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: Container(
            padding: const EdgeInsets.fromLTRB(24, 48, 24, 56),
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.bottomCenter,
                end: Alignment.topCenter,
                colors: [Colors.black87, Colors.transparent],
                stops: [0.55, 1.0],
              ),
            ),
            child: _capturing
                ? Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const CircularProgressIndicator(color: Colors.white),
                      const SizedBox(height: 12),
                      Text('Wysyłanie…',
                          style:
                              tt.bodyMedium?.copyWith(color: Colors.white70)),
                    ],
                  )
                : Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        (_remainingMs / 1000.0).toStringAsFixed(1),
                        style: const TextStyle(
                          fontSize: 100,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                          height: 1.0,
                          letterSpacing: -4,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Przygotuj szachownicę',
                        style: tt.titleMedium?.copyWith(color: Colors.white70),
                      ),
                    ],
                  ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final tt = theme.textTheme;
    final session = state.session;

    final minFrames = session?.minCalibFrames ?? 0;
    final canCalibrate = state.isLeader && minFrames >= 3;
    final busy = _capturing || state.isLoading || !state.wsConnected;
    final isCountingDown = _remainingMs > 0;
    final showCapturePanel = isCountingDown || _capturing;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Kalibracja'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_outlined),
            tooltip: 'Odśwież',
            onPressed: () => state.refreshSession(),
          ),
        ],
      ),
      body: showCapturePanel
          ? _buildCameraView(tt)
          : ListView(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              children: [
                AppBanners(
                  error: state.error,
                  info: state.info,
                  onClearError: state.clearError,
                  onClearInfo: state.clearInfo,
                ),
            // Instructions
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Instrukcja',
                        style: tt.titleSmall?.copyWith(
                            color: cs.primary, fontWeight: FontWeight.w600)),
                    const SizedBox(height: 8),
                    Text(
                      '1. Ustaw szachownicę tak, żeby była widoczna z obu kamer.\n'
                      '2. Lider klika „Zrób klatkę" — oba telefony fotografują\n'
                      '   jednocześnie po 3-sekundowym odliczaniu.\n'
                      '3. Powtórz 10–15 razy zmieniając kąt ustawienia szachownicy.\n'
                      '4. Gdy każde urządzenie ma ≥ 3 klatki, lider uruchamia kalibrację.',
                      style: tt.bodySmall
                          ?.copyWith(color: cs.onSurfaceVariant, height: 1.5),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 12),

            // Progress per device
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text('Postęp kalibracji',
                            style: tt.titleSmall?.copyWith(
                                color: cs.primary,
                                fontWeight: FontWeight.w600)),
                        const Spacer(),
                        if (session != null &&
                            session.devices.any(
                                (d) => d.isCamera && d.calibFrameCount > 0))
                          TextButton.icon(
                            onPressed: () => Navigator.push(
                              context,
                              MaterialPageRoute(
                                  builder: (_) => const CalibImagesScreen()),
                            ),
                            icon: const Icon(Icons.photo_library_outlined,
                                size: 16),
                            label: const Text('Przeglądaj'),
                            style: TextButton.styleFrom(
                                padding: EdgeInsets.zero,
                                visualDensity: VisualDensity.compact),
                          ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    if (session != null)
                      ...session.devices.map(
                        (d) => d.isCamera
                            ? Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Row(
                                    children: [
                                      ConnectionDot(
                                        active: d.deviceId == state.deviceId
                                            ? state.wsConnected
                                            : d.wsConnected,
                                        activeColor: AppColors.success,
                                        inactiveColor: cs.outline,
                                      ),
                                      const SizedBox(width: 4),
                                      Text(
                                        d.deviceId,
                                        style: tt.bodySmall?.copyWith(
                                          color: d.isLeader ? cs.primary : null,
                                          fontWeight: d.isLeader ? FontWeight.w600 : null,
                                        ),
                                      ),
                                      const Spacer(),
                                      Text('${d.calibFrameCount} klatek',
                                          style: tt.bodySmall?.copyWith(
                                              color: cs.onSurfaceVariant)),
                                    ],
                                  ),
                                  const SizedBox(height: 6),
                                  ClipRRect(
                                    borderRadius: BorderRadius.circular(4),
                                    child: LinearProgressIndicator(
                                      value: (d.calibFrameCount / 10.0)
                                          .clamp(0.0, 1.0),
                                      minHeight: 6,
                                      backgroundColor:
                                          cs.surfaceContainerHighest,
                                    ),
                                  ),
                                  const SizedBox(height: 10),
                                ],
                              )
                            : Padding(
                                padding:
                                    const EdgeInsets.only(bottom: 10),
                                child: Row(
                                  children: [
                                    Icon(Icons.computer,
                                        size: 14,
                                        color: cs.onSurfaceVariant),
                                    const SizedBox(width: 8),
                                    Text(d.deviceId,
                                        style: tt.bodySmall?.copyWith(
                                            color: cs.onSurfaceVariant)),
                                    const Spacer(),
                                    Text('admin',
                                        style: tt.bodySmall?.copyWith(
                                            color: cs.onSurfaceVariant)),
                                  ],
                                ),
                              ),
                      ),
                    Text(
                      'Minimum klatek (wszystkie urządzenia): $minFrames',
                      style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 12),

            // WS connectivity warning
            if (!state.wsConnected) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 10),
                  child: Row(
                    children: [
                      Icon(Icons.wifi_off, size: 16, color: cs.error),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'WebSocket rozłączony — poczekaj na reconnect',
                          style: tt.bodySmall?.copyWith(color: cs.error),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 8),
            ],

            // Synchronized trigger (leader) or waiting card (follower)
            if (state.isLeader)
              ElevatedButton.icon(
                onPressed: busy ? null : () => state.triggerCalibCapture(delayMs: 3000),
                icon: const Icon(Icons.camera_alt_outlined),
                label: const Text('Zrób klatkę kalibracyjną (3 s odliczanie)'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.stateProcessing,
                  foregroundColor: Colors.white,
                ),
              )
            else
              Card(
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 12),
                  child: Row(
                    children: [
                      Icon(Icons.info_outline,
                          size: 16, color: cs.onSurfaceVariant),
                      const SizedBox(width: 8),
                      Text('Oczekiwanie na trigger lidera…',
                          style: tt.bodySmall
                              ?.copyWith(color: cs.onSurfaceVariant)),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 8),

            // Manual single-shot fallback
            OutlinedButton.icon(
              onPressed: busy ? null : _manualCapture,
              icon: const Icon(Icons.add_a_photo_outlined),
              label: const Text('Ręczne zdjęcie'),
            ),

            const SizedBox(height: 12),

            // Start calibration (leader only)
            if (state.isLeader)
              ElevatedButton.icon(
                onPressed: (canCalibrate && !state.isLoading)
                    ? () => _startCalibration(state)
                    : null,
                icon: state.isLoading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.calculate_outlined),
                label: Text(
                  canCalibrate
                      ? 'Uruchom kalibrację'
                      : 'Potrzeba min. 3 klatek na urządzenie',
                ),
                style: canCalibrate
                    ? ElevatedButton.styleFrom(
                        backgroundColor: AppColors.stateDone,
                        foregroundColor: Colors.white,
                      )
                    : null,
              )
            else
              Card(
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 12),
                  child: Row(
                    children: [
                      Icon(Icons.info_outline,
                          size: 16, color: cs.onSurfaceVariant),
                      const SizedBox(width: 8),
                      Text('Kalibrację uruchamia lider sesji',
                          style: tt.bodySmall
                              ?.copyWith(color: cs.onSurfaceVariant)),
                    ],
                  ),
                ),
              ),

                const SizedBox(height: 8),
              ],
            ),
    );
  }
}
