import 'dart:async';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';
import '../theme/app_theme.dart';
import '../utils/log.dart';
import '../widgets/app_banner.dart';

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  static const _log = Log('CaptureScreen');

  final _picker = ImagePicker();
  late final AppState _appState;
  bool _uploading = false;
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
    if (_appState.isCamera && _appState.captureTriggerAt != null) {
      _startCountdown();
    }
  }

  void _checkExistingTrigger() {
    // Po pierwszej klatce - _startCountdown wywołuje setState, więc nie może
    // zostać wywołane synchronicznie w initState.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted && _appState.isCamera && _appState.captureTriggerAt != null) {
        _startCountdown();
      }
    });
  }

  void _startCountdown() {
    _countdownTimer?.cancel();
    final triggerAt = _appState.captureTriggerAt!;
    final offset = _appState.serverTimeOffset;
    _appState.clearCaptureTrigger();

    int remainingMs() {
      final now = DateTime.now().millisecondsSinceEpoch / 1000.0;
      return ((triggerAt - now - offset) * 1000).round();
    }

    // Ignoruj nieaktualne wyzwolenia (np. odebrane gdy ekran był zamknięty) -
    // nie otwieramy automatycznie aparatu dla momentu, który już minął.
    if (remainingMs() <= 0) {
      _log.warn('Pominięto nieaktualny trigger przechwycenia '
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

  Future<void> _triggerCapture() async {
    await _appState.triggerCapture(delayMs: 3000);
  }

  Future<void> _captureAndUpload() async {
    XFile? xfile;
    try {
      if (kIsWeb) {
        xfile = await _picker.pickImage(source: ImageSource.gallery);
      } else {
        xfile = await _picker.pickImage(
          source: ImageSource.camera,
          imageQuality: 90,
        );
      }
    } catch (e, st) {
      _log.warn('Błąd aparatu przy przechwytywaniu zdjęcia (kIsWeb=$kIsWeb)', e, st);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Błąd aparatu: $e')),
        );
      }
      return;
    }
    if (xfile == null) {
      _log.info('Przechwytywanie anulowane przez użytkownika (brak pliku)');
      return;
    }

    setState(() => _uploading = true);
    final bytes = await xfile.readAsBytes();
    final ok = await _appState.uploadCaptureImage(bytes);
    if (mounted) {
      setState(() => _uploading = false);
      if (ok) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Zdjęcie przesłane')),
        );
      }
    }
  }

  Future<void> _startMeasurement() async {
    await _appState.startMeasurement();
    if (mounted && _appState.error == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Pipeline pomiaru uruchomiony — czekaj na wynik')),
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

    final myCaptures = state.myDevice?.captureFrameCount ?? 0;

    final allCaptured = session?.allCaptured ?? false;
    final isCountingDown = _remainingMs > 0;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Przechwycenie'),
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
                    Text('Przygotuj się',
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
            // Device capture status
            if (session != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Status urządzeń',
                          style: tt.titleSmall?.copyWith(
                              color: cs.primary, fontWeight: FontWeight.w600)),
                      const SizedBox(height: 12),
                      ...session.devices.map(
                        (d) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          child: Row(
                            children: [
                              Container(
                                width: 6,
                                height: 6,
                                decoration: BoxDecoration(
                                  color: d.captureFrameCount > 0
                                      ? AppColors.success
                                      : cs.outline,
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(d.deviceId, style: tt.bodySmall),
                              ),
                              Text('${d.captureFrameCount} zdjęć',
                                  style: tt.bodySmall
                                      ?.copyWith(color: cs.onSurfaceVariant)),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 12),

            // Trigger (leader only)
            if (state.isLeader)
              ElevatedButton.icon(
                onPressed: state.isLoading ? null : _triggerCapture,
                icon: const Icon(Icons.flash_on_outlined),
                label: const Text('Wyzwól przechwycenie (3 s odliczanie)'),
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
                      Text('Przechwycenie wyzwala lider sesji',
                          style: tt.bodySmall
                              ?.copyWith(color: cs.onSurfaceVariant)),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 8),

            // Manual capture / upload button
            ElevatedButton.icon(
              onPressed:
                  (_uploading || state.isLoading) ? null : _captureAndUpload,
              icon: _uploading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(kIsWeb
                      ? Icons.upload_file_outlined
                      : Icons.camera_alt_outlined),
              label: Text(
                kIsWeb
                    ? 'Prześlij zdjęcie pomiaru'
                    : 'Zrób zdjęcie ($myCaptures wykonanych)',
              ),
            ),

            const SizedBox(height: 12),

            // Start measurement (leader, when all captured)
            if (state.isLeader)
              ElevatedButton.icon(
                onPressed: (allCaptured && !state.isLoading)
                    ? _startMeasurement
                    : null,
                icon: state.isLoading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.straighten_outlined),
                label: Text(
                  allCaptured
                      ? 'Uruchom pomiar'
                      : 'Poczekaj aż wszystkie urządzenia wykonają zdjęcie',
                ),
                style: allCaptured
                    ? ElevatedButton.styleFrom(
                        backgroundColor: AppColors.stateDone,
                        foregroundColor: Colors.white,
                      )
                    : null,
              ),
          ],
        ],
      ),
    );
  }
}
