import 'dart:async';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';
import '../theme/app_theme.dart';
import '../utils/log.dart';
import '../widgets/app_banner.dart';

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
    super.dispose();
  }

  void _onStateChange() {
    if (_appState.calibTriggerAt != null) {
      _startCountdown();
    }
  }

  void _checkExistingTrigger() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && _appState.calibTriggerAt != null) {
        _startCountdown();
      }
    });
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
    _countdownTimer = Timer.periodic(const Duration(milliseconds: 50), (t) {
      if (!mounted) {
        t.cancel();
        return;
      }
      final remaining = remainingMs();
      if (remaining <= 0) {
        t.cancel();
        setState(() => _remainingMs = 0);
        _captureAndUpload();
      } else {
        setState(() => _remainingMs = remaining);
      }
    });
  }

  Future<void> _captureAndUpload() async {
    if (_capturing) return;
    setState(() => _capturing = true);
    try {
      final xfile = await _picker.pickImage(
        source: kIsWeb ? ImageSource.gallery : ImageSource.camera,
        imageQuality: 90,
      );
      if (xfile == null) {
        _log.info('Zdjęcie kalibracyjne anulowane przez użytkownika');
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
      _log.warn('Błąd podczas przechwytywania klatki kalibracyjnej', e, st);
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

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);
    final cs = theme.colorScheme;
    final tt = theme.textTheme;
    final session = state.session;

    final minFrames = session?.minCalibFrames ?? 0;
    final canCalibrate = state.isLeader && minFrames >= 3;
    final busy = _capturing || state.isLoading;
    final isCountingDown = _remainingMs > 0;

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
      body: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        children: [
          AppBanners(
            error: state.error,
            info: state.info,
            onClearError: state.clearError,
            onClearInfo: state.clearInfo,
          ),

          // Countdown display
          if (isCountingDown)
            Card(
              child: Padding(
                padding:
                    const EdgeInsets.symmetric(vertical: 24, horizontal: 16),
                child: Column(
                  children: [
                    Text('Przygotuj szachownicę',
                        style: tt.titleMedium
                            ?.copyWith(fontWeight: FontWeight.w600)),
                    const SizedBox(height: 12),
                    Text(
                      '${(_remainingMs / 1000.0).toStringAsFixed(1)} s',
                      style: const TextStyle(
                        fontSize: 64,
                        fontWeight: FontWeight.bold,
                        color: AppColors.warning,
                        letterSpacing: -2,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text('Aparat zostanie uruchomiony automatycznie',
                        style:
                            tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
                  ],
                ),
              ),
            ),

          if (!isCountingDown) ...[
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
                    Text('Postęp kalibracji',
                        style: tt.titleSmall?.copyWith(
                            color: cs.primary, fontWeight: FontWeight.w600)),
                    const SizedBox(height: 12),
                    if (session != null)
                      ...session.devices.map(
                        (d) => Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Container(
                                  width: 6,
                                  height: 6,
                                  decoration: BoxDecoration(
                                    color: d.isLeader
                                        ? AppColors.stateReady
                                        : cs.onSurfaceVariant,
                                    shape: BoxShape.circle,
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Text(d.deviceId, style: tt.bodySmall),
                                const Spacer(),
                                Text('${d.calibFrameCount} klatek',
                                    style: tt.bodySmall
                                        ?.copyWith(color: cs.onSurfaceVariant)),
                              ],
                            ),
                            const SizedBox(height: 6),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(4),
                              child: LinearProgressIndicator(
                                value:
                                    (d.calibFrameCount / 10.0).clamp(0.0, 1.0),
                                minHeight: 6,
                                backgroundColor: cs.surfaceContainerHighest,
                              ),
                            ),
                            const SizedBox(height: 10),
                          ],
                        ),
                      ),
                    Text(
                      'Minimum klatek (wszystkie urządzenia): $minFrames',
                      style:
                          tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 12),

            // Synchronized trigger (leader) or waiting card (follower)
            if (state.isLeader)
              ElevatedButton.icon(
                onPressed: busy ? null : () => state.triggerCalibCapture(delayMs: 3000),
                icon: busy
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.camera_alt_outlined),
                label: const Text('Zrób klatkę kalibracyjną (3 s odliczanie)'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.stateProcessing,
                  foregroundColor: Colors.white,
                ),
              )
            else
              Card(
                child: Padding(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  child: Row(
                    children: [
                      Icon(Icons.info_outline,
                          size: 16, color: cs.onSurfaceVariant),
                      const SizedBox(width: 8),
                      Text('Oczekiwanie na trigger lidera…',
                          style:
                              tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 8),

            // Manual single-shot fallback
            OutlinedButton.icon(
              onPressed: busy ? null : _captureAndUpload,
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
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  child: Row(
                    children: [
                      Icon(Icons.info_outline,
                          size: 16, color: cs.onSurfaceVariant),
                      const SizedBox(width: 8),
                      Text('Kalibrację uruchamia lider sesji',
                          style:
                              tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 8),
          ],
        ],
      ),
    );
  }
}
