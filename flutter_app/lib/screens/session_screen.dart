import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/app_state.dart';
import 'calibration_screen.dart';
import 'capture_screen.dart';
import 'results_screen.dart';

class SessionScreen extends StatelessWidget {
  const SessionScreen({super.key});

  Color _stateColor(String state, ColorScheme cs) {
    switch (state) {
      case 'IDLE':
        return Colors.grey;
      case 'CALIBRATING':
        return Colors.orange;
      case 'READY':
        return cs.primary;
      case 'PROCESSING':
        return Colors.purple;
      case 'DONE':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  String _stateLabel(String state) {
    switch (state) {
      case 'IDLE':
        return 'Oczekiwanie';
      case 'CALIBRATING':
        return 'Kalibracja';
      case 'READY':
        return 'Gotowa';
      case 'PROCESSING':
        return 'Przetwarzanie';
      case 'DONE':
        return 'Zakończona';
      default:
        return state;
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);
    final session = state.session;

    return Scaffold(
      appBar: AppBar(
        title: Text(session == null ? 'Sesja' : 'Sesja ${session.sessionId.substring(0, 8)}…'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Odśwież',
            onPressed: () => state.refreshSession(),
          ),
          IconButton(
            icon: const Icon(Icons.exit_to_app),
            tooltip: 'Opuść sesję',
            onPressed: () async {
              await state.leaveSession();
              if (context.mounted) Navigator.pop(context);
            },
          ),
        ],
      ),
      body: session == null
          ? const Center(child: CircularProgressIndicator())
          : _SessionBody(session: session, appState: state, theme: theme),
    );
  }
}

class _SessionBody extends StatelessWidget {
  final SessionData session;
  final AppState appState;
  final ThemeData theme;

  const _SessionBody({
    required this.session,
    required this.appState,
    required this.theme,
  });

  @override
  Widget build(BuildContext context) {
    final stateColor = _stateColor(session.state, theme.colorScheme);

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Error / info banners
        if (appState.error != null)
          _Banner(
            color: theme.colorScheme.errorContainer,
            textColor: theme.colorScheme.onErrorContainer,
            icon: Icons.error_outline,
            message: appState.error!,
            onClose: appState.clearError,
          ),
        if (appState.info != null)
          _Banner(
            color: theme.colorScheme.primaryContainer,
            textColor: theme.colorScheme.onPrimaryContainer,
            icon: Icons.info_outline,
            message: appState.info!,
            onClose: appState.clearInfo,
          ),

        // Session state chip
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Icon(Icons.circle, color: stateColor, size: 14),
                const SizedBox(width: 8),
                Text(
                  _stateLabel(session.state),
                  style: theme.textTheme.titleMedium?.copyWith(color: stateColor),
                ),
                const Spacer(),
                Text(
                  'ID: ${session.sessionId.substring(0, 8)}…',
                  style: theme.textTheme.bodySmall,
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),

        // WS connection status
        Row(
          children: [
            Icon(
              Icons.wifi,
              size: 16,
              color: appState.serverTimeOffset != 0.0 ? Colors.green : Colors.grey,
            ),
            const SizedBox(width: 4),
            Text(
              appState.serverTimeOffset != 0.0
                  ? 'WebSocket połączony (offset: ${appState.serverTimeOffset.toStringAsFixed(3)} s)'
                  : 'WebSocket rozłączony',
              style: theme.textTheme.bodySmall,
            ),
            const Spacer(),
            TextButton(
              onPressed: () => appState.reconnectWs(),
              child: const Text('Reconnect'),
            ),
          ],
        ),
        const SizedBox(height: 8),

        // Device list
        Text('Urządzenia (${session.devices.length})',
            style: theme.textTheme.titleSmall),
        const SizedBox(height: 6),
        ...session.devices.map((d) => _DeviceCard(device: d, theme: theme)),

        const SizedBox(height: 16),

        // Navigation buttons
        Text('Akcje', style: theme.textTheme.titleSmall),
        const SizedBox(height: 8),

        // Calibration
        _ActionButton(
          label: 'Kalibracja',
          icon: Icons.camera_enhance,
          enabled: session.isIdle || session.isCalibrating || session.isReady,
          onTap: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const CalibrationScreen()),
          ),
        ),
        const SizedBox(height: 8),

        // Capture
        _ActionButton(
          label: 'Przechwycenie',
          icon: Icons.photo_camera,
          enabled: session.isReady || session.isProcessing || session.isDone,
          onTap: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const CaptureScreen()),
          ),
        ),
        const SizedBox(height: 8),

        // Results
        _ActionButton(
          label: 'Wyniki',
          icon: Icons.analytics,
          enabled: session.isDone || appState.measurement != null,
          onTap: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const ResultsScreen()),
          ),
        ),

        const SizedBox(height: 24),

        // WS event log
        ExpansionTile(
          title: const Text('Log zdarzeń WebSocket'),
          children: [
            if (appState.wsLog.isEmpty)
              const Padding(
                padding: EdgeInsets.all(12),
                child: Text('Brak zdarzeń'),
              )
            else
              ...appState.wsLog.reversed.take(15).map(
                    (e) => Padding(
                      padding:
                          const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
                      child: Text(
                        e.toString(),
                        style: theme.textTheme.bodySmall
                            ?.copyWith(fontFamily: 'monospace'),
                      ),
                    ),
                  ),
          ],
        ),
      ],
    );
  }

  Color _stateColor(String state, ColorScheme cs) {
    switch (state) {
      case 'IDLE':
        return Colors.grey;
      case 'CALIBRATING':
        return Colors.orange;
      case 'READY':
        return cs.primary;
      case 'PROCESSING':
        return Colors.purple;
      case 'DONE':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  String _stateLabel(String state) {
    switch (state) {
      case 'IDLE':
        return 'Oczekiwanie';
      case 'CALIBRATING':
        return 'Kalibracja';
      case 'READY':
        return 'Gotowa';
      case 'PROCESSING':
        return 'Przetwarzanie';
      case 'DONE':
        return 'Zakończona';
      default:
        return state;
    }
  }
}

class _DeviceCard extends StatelessWidget {
  final DeviceInfo device;
  final ThemeData theme;

  const _DeviceCard({required this.device, required this.theme});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: ListTile(
        leading: Icon(
          device.isLeader ? Icons.star : Icons.phone_android,
          color: device.isLeader ? Colors.amber : theme.colorScheme.primary,
        ),
        title: Text(device.deviceId),
        subtitle: Text(device.mac),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            // WS connected
            Icon(
              Icons.wifi,
              size: 16,
              color: device.wsConnected ? Colors.green : Colors.grey,
            ),
            const SizedBox(width: 8),
            // Frame counts
            Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text('Kalib: ${device.calibFrameCount}',
                    style: theme.textTheme.bodySmall),
                Text('Cap: ${device.captureFrameCount}',
                    style: theme.textTheme.bodySmall),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool enabled;
  final VoidCallback onTap;

  const _ActionButton({
    required this.label,
    required this.icon,
    required this.enabled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      onPressed: enabled ? onTap : null,
      icon: Icon(icon),
      label: Text(label),
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
