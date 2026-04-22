import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/app_state.dart';
import '../theme/app_theme.dart';
import '../widgets/app_banner.dart';
import 'calibration_screen.dart';
import 'capture_screen.dart';
import 'results_screen.dart';

class SessionScreen extends StatelessWidget {
  const SessionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state   = context.watch<AppState>();
    final theme   = Theme.of(context);
    final session = state.session;

    return Scaffold(
      appBar: AppBar(
        title: Text(session == null
            ? 'Sesja'
            : 'Sesja  ${session.sessionId.substring(0, 8)}…'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_outlined),
            tooltip: 'Odśwież',
            onPressed: () => state.refreshSession(),
          ),
          IconButton(
            icon: const Icon(Icons.logout_outlined),
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
  final AppState    appState;
  final ThemeData   theme;

  const _SessionBody({
    required this.session,
    required this.appState,
    required this.theme,
  });

  @override
  Widget build(BuildContext context) {
    final cs    = theme.colorScheme;
    final tt    = theme.textTheme;
    final color = stateColor(session.state, cs);

    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      children: [
        if (appState.error != null)
          AppBanner(
            color: cs.errorContainer,
            textColor: cs.onErrorContainer,
            icon: Icons.error_outline,
            message: appState.error!,
            onClose: appState.clearError,
          ),
        if (appState.info != null)
          AppBanner(
            color: cs.secondaryContainer,
            textColor: cs.onSecondaryContainer,
            icon: Icons.info_outline,
            message: appState.info!,
            onClose: appState.clearInfo,
          ),

        // State card
        Card(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(color: color, shape: BoxShape.circle),
                ),
                const SizedBox(width: 10),
                Text(stateLabel(session.state),
                    style: tt.titleSmall?.copyWith(
                        color: color, fontWeight: FontWeight.w600)),
                const Spacer(),
                Text('ID: ${session.sessionId.substring(0, 8)}…',
                    style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),

        // WebSocket status row
        _WsStatusRow(appState: appState, cs: cs, tt: tt),
        const SizedBox(height: 12),

        // Devices
        Text('Urządzenia (${session.devices.length})',
            style: tt.labelMedium?.copyWith(
                color: cs.onSurfaceVariant, letterSpacing: 0.3)),
        const SizedBox(height: 6),
        ...session.devices.map((d) => _DeviceCard(device: d, cs: cs, tt: tt)),

        const SizedBox(height: 16),

        // Actions
        Text('Nawigacja',
            style: tt.labelMedium?.copyWith(
                color: cs.onSurfaceVariant, letterSpacing: 0.3)),
        const SizedBox(height: 8),

        _NavTile(
          icon: Icons.tune_outlined,
          label: 'Kalibracja',
          subtitle: 'Zdjęcia szachownicy i obliczenia stereo',
          enabled: session.isIdle || session.isCalibrating || session.isReady,
          onTap: () => Navigator.push(
              context, MaterialPageRoute(builder: (_) => const CalibrationScreen())),
        ),
        _NavTile(
          icon: Icons.camera_outlined,
          label: 'Przechwycenie',
          subtitle: 'Synchroniczne zdjęcie pomiarowe',
          enabled: session.isReady || session.isProcessing || session.isDone,
          onTap: () => Navigator.push(
              context, MaterialPageRoute(builder: (_) => const CaptureScreen())),
        ),
        _NavTile(
          icon: Icons.bar_chart_outlined,
          label: 'Wyniki',
          subtitle: 'Wymiary i raport pomiaru',
          enabled: session.isDone || appState.measurement != null,
          onTap: () => Navigator.push(
              context, MaterialPageRoute(builder: (_) => const ResultsScreen())),
        ),

        const SizedBox(height: 20),

        // WS log
        ExpansionTile(
          tilePadding: EdgeInsets.zero,
          title: Text('Dziennik WebSocket',
              style: tt.labelMedium?.copyWith(color: cs.onSurfaceVariant)),
          children: [
            Container(
              decoration: BoxDecoration(
                color: cs.surfaceContainerLowest,
                border: Border.all(color: cs.outlineVariant.withOpacity(0.5)),
                borderRadius: BorderRadius.circular(8),
              ),
              padding: const EdgeInsets.all(10),
              child: appState.wsLog.isEmpty
                  ? Text('Brak zdarzeń', style: tt.bodySmall)
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: appState.wsLog.reversed.take(15).map((e) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 1),
                        child: Text(e.toString(),
                            style: tt.bodySmall?.copyWith(
                                fontFamily: 'monospace',
                                color: cs.onSurfaceVariant)),
                      )).toList(),
                    ),
            ),
          ],
        ),
        const SizedBox(height: 8),
      ],
    );
  }
}

class _WsStatusRow extends StatelessWidget {
  final AppState      appState;
  final ColorScheme   cs;
  final TextTheme     tt;

  const _WsStatusRow({
    required this.appState,
    required this.cs,
    required this.tt,
  });

  @override
  Widget build(BuildContext context) {
    final connected = appState.wsConnected;
    return Row(
      children: [
        Container(
          width: 6, height: 6,
          decoration: BoxDecoration(
            color: connected ? AppColors.success : cs.outline,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 8),
        Text(
          connected
              ? 'WebSocket  ·  offset ${appState.serverTimeOffset.toStringAsFixed(3)} s'
              : 'WebSocket rozłączony',
          style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
        ),
        const Spacer(),
        TextButton(
          onPressed: appState.reconnectWs,
          child: const Text('Połącz ponownie'),
        ),
      ],
    );
  }
}

class _DeviceCard extends StatelessWidget {
  final DeviceInfo  device;
  final ColorScheme cs;
  final TextTheme   tt;

  const _DeviceCard({required this.device, required this.cs, required this.tt});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: ListTile(
        leading: Container(
          width: 36, height: 36,
          decoration: BoxDecoration(
            color: device.isLeader
                ? AppColors.stateReady.withOpacity(0.12)
                : cs.surfaceContainerHighest,
            shape: BoxShape.circle,
          ),
          child: Icon(
            device.isLeader ? Icons.account_circle_outlined : Icons.smartphone_outlined,
            size: 20,
            color: device.isLeader ? AppColors.stateReady : cs.onSurfaceVariant,
          ),
        ),
        title: Text(device.deviceId, style: tt.bodyMedium),
        subtitle: Text(device.mac,
            style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 6, height: 6,
              decoration: BoxDecoration(
                color: device.wsConnected ? AppColors.success : cs.outline,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 10),
            Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text('Kal. ${device.calibFrameCount}',
                    style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
                Text('Cap. ${device.captureFrameCount}',
                    style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _NavTile extends StatelessWidget {
  final IconData icon;
  final String   label;
  final String   subtitle;
  final bool     enabled;
  final VoidCallback onTap;

  const _NavTile({
    required this.icon,
    required this.label,
    required this.subtitle,
    required this.enabled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: ListTile(
        leading: Container(
          width: 36, height: 36,
          decoration: BoxDecoration(
            color: enabled
                ? cs.primaryContainer.withOpacity(0.6)
                : cs.surfaceContainerHighest,
            shape: BoxShape.circle,
          ),
          child: Icon(icon, size: 18,
              color: enabled ? cs.primary : cs.onSurfaceVariant),
        ),
        title: Text(label,
            style: tt.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
                color: enabled ? cs.onSurface : cs.onSurfaceVariant)),
        subtitle: Text(subtitle, style: tt.bodySmall),
        trailing: enabled
            ? Icon(Icons.chevron_right, color: cs.onSurfaceVariant)
            : null,
        onTap: enabled ? onTap : null,
      ),
    );
  }
}
