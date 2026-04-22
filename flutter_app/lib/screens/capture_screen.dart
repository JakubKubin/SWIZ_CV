import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  final _picker = ImagePicker();
  bool _uploading = false;
  Timer? _countdownTimer;
  int _remainingMs = 0;

  @override
  void initState() {
    super.initState();
    final appState = context.read<AppState>();
    appState.addListener(_onStateChange);
    _checkExistingTrigger(appState);
  }

  @override
  void dispose() {
    context.read<AppState>().removeListener(_onStateChange);
    _countdownTimer?.cancel();
    super.dispose();
  }

  void _onStateChange() {
    final appState = context.read<AppState>();
    if (appState.captureTriggerAt != null) {
      _startCountdown(appState);
    }
  }

  void _checkExistingTrigger(AppState appState) {
    if (appState.captureTriggerAt != null) {
      _startCountdown(appState);
    }
  }

  void _startCountdown(AppState appState) {
    _countdownTimer?.cancel();
    final triggerAt = appState.captureTriggerAt!;
    appState.clearCaptureTrigger();

    _countdownTimer = Timer.periodic(const Duration(milliseconds: 50), (t) {
      final now = DateTime.now().millisecondsSinceEpoch / 1000.0;
      final remaining = ((triggerAt - now) * 1000).round();
      if (remaining <= 0) {
        t.cancel();
        setState(() => _remainingMs = 0);
        // Auto-launch camera after countdown
        _captureAndUpload(context.read<AppState>());
      } else {
        setState(() => _remainingMs = remaining);
      }
    });
  }

  Future<void> _triggerCapture(AppState state) async {
    await state.triggerCapture(delayMs: 3000);
  }

  Future<void> _captureAndUpload(AppState state) async {
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
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Błąd aparatu: $e')),
        );
      }
      return;
    }
    if (xfile == null) return;

    setState(() => _uploading = true);
    final bytes = await xfile.readAsBytes();
    final ok = await state.uploadCaptureImage(Uint8List.fromList(bytes));
    if (mounted) {
      setState(() => _uploading = false);
      if (ok) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Zdjęcie przesłane'),
            backgroundColor: Colors.green,
          ),
        );
      }
    }
  }

  Future<void> _startMeasurement(AppState state) async {
    await state.startMeasurement();
    if (mounted && state.error == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Pipeline pomiaru uruchomiony — czekaj na wynik')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);
    final session = state.session;

    final myCaptures = session?.devices
            .where((d) => d.deviceId == state.deviceId)
            .firstOrNull
            ?.captureFrameCount ??
        0;

    final allCaptured = session?.allCaptured ?? false;
    final isCountingDown = _remainingMs > 0;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Przechwycenie'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => state.refreshSession(),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Error / info banners
          if (state.error != null)
            _Banner(
              color: theme.colorScheme.errorContainer,
              textColor: theme.colorScheme.onErrorContainer,
              icon: Icons.error_outline,
              message: state.error!,
              onClose: state.clearError,
            ),
          if (state.info != null)
            _Banner(
              color: theme.colorScheme.primaryContainer,
              textColor: theme.colorScheme.onPrimaryContainer,
              icon: Icons.info_outline,
              message: state.info!,
              onClose: state.clearInfo,
            ),

          // Countdown display
          if (isCountingDown)
            Card(
              color: Colors.deepOrange.shade50,
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  children: [
                    const Text(
                      'Przygotuj się!',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      '${(_remainingMs / 1000.0).toStringAsFixed(1)} s',
                      style: TextStyle(
                        fontSize: 64,
                        fontWeight: FontWeight.bold,
                        color: Colors.deepOrange.shade700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text('Aparat zostanie uruchomiony automatycznie'),
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
                      Text('Status urządzeń', style: theme.textTheme.titleMedium),
                      const SizedBox(height: 12),
                      ...session.devices.map(
                        (d) => ListTile(
                          contentPadding: EdgeInsets.zero,
                          leading: Icon(
                            d.captureFrameCount > 0
                                ? Icons.check_circle
                                : Icons.radio_button_unchecked,
                            color: d.captureFrameCount > 0 ? Colors.green : Colors.grey,
                          ),
                          title: Text(d.deviceId),
                          trailing: Text('${d.captureFrameCount} zdjęć'),
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
                onPressed: state.isLoading ? null : () => _triggerCapture(state),
                icon: const Icon(Icons.flash_on),
                label: const Text('Wyzwól przechwycenie (3 s odliczanie)'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepOrange,
                  foregroundColor: Colors.white,
                ),
              )
            else
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Row(
                    children: const [
                      Icon(Icons.info_outline, size: 18),
                      SizedBox(width: 8),
                      Text('Przechwycenie wyzwala lider sesji'),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 8),

            // Manual capture/upload button
            ElevatedButton.icon(
              onPressed: (_uploading || state.isLoading)
                  ? null
                  : () => _captureAndUpload(state),
              icon: _uploading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : Icon(kIsWeb ? Icons.upload_file : Icons.camera_alt),
              label: Text(
                kIsWeb ? 'Prześlij zdjęcie pomiaru' : 'Zrób zdjęcie ($myCaptures wykonanych)',
              ),
            ),

            const SizedBox(height: 12),

            // Start measurement (leader, when all captured)
            if (state.isLeader)
              ElevatedButton.icon(
                onPressed: (allCaptured && !state.isLoading)
                    ? () => _startMeasurement(state)
                    : null,
                icon: state.isLoading
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.straighten),
                label: Text(
                  allCaptured
                      ? 'Uruchom pomiar'
                      : 'Poczekaj aż wszystkie urządzenia wykonają zdjęcie',
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: allCaptured ? Colors.green : null,
                  foregroundColor: allCaptured ? Colors.white : null,
                ),
              ),
          ],
        ],
      ),
    );
  }
}

class _Banner extends StatelessWidget {
  final Color color;
  final Color textColor;
  final IconData icon;
  final String message;
  final VoidCallback onClose;

  const _Banner({
    required this.color,
    required this.textColor,
    required this.icon,
    required this.message,
    required this.onClose,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      color: color,
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: Row(
          children: [
            Icon(icon, color: textColor, size: 20),
            const SizedBox(width: 8),
            Expanded(child: Text(message, style: TextStyle(color: textColor))),
            IconButton(
              icon: Icon(Icons.close, color: textColor, size: 18),
              onPressed: onClose,
              padding: EdgeInsets.zero,
              constraints: const BoxConstraints(),
            ),
          ],
        ),
      ),
    );
  }
}
