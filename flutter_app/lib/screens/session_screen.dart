import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/app_state.dart';
import '../theme/app_theme.dart';
import '../widgets/app_banner.dart';
import '../widgets/connection_dot.dart';
import 'calibration_screen.dart';
import 'capture_screen.dart';
import 'results_screen.dart';

class SessionScreen extends StatefulWidget {
  const SessionScreen({super.key});

  @override
  State<SessionScreen> createState() => _SessionScreenState();
}

class _SessionScreenState extends State<SessionScreen> {
  late final AppState _appState;

  @override
  void initState() {
    super.initState();
    _appState = context.read<AppState>();
    _appState.registerNavigation(
      toCalib: _autoNavigateToCalib,
      toCapture: _autoNavigateToCapture,
    );
  }

  @override
  void dispose() {
    _appState.unregisterNavigation();
    super.dispose();
  }

  void _autoNavigateToCalib() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      // If CalibrationScreen was already active its _onStateChange cleared the
      // trigger during notifyListeners() — skip navigation in that case.
      if (context.read<AppState>().calibTriggerAt == null) return;
      Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const CalibrationScreen()),
      );
    });
  }

  void _autoNavigateToCapture() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      if (context.read<AppState>().captureTriggerAt == null) return;
      Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const CaptureScreen()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);
    final session = state.session;

    return Scaffold(
      appBar: AppBar(
        title: Text(session == null ? 'Sesja' : 'Sesja ${session.sessionId}'),
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
  final AppState appState;
  final ThemeData theme;

  const _SessionBody({
    required this.session,
    required this.appState,
    required this.theme,
  });

  @override
  Widget build(BuildContext context) {
    final cs = theme.colorScheme;
    final tt = theme.textTheme;
    final color = stateColor(session.state, cs);

    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      children: [
        AppBanners(
          error: appState.error,
          info: appState.info,
          onClearError: appState.clearError,
          onClearInfo: appState.clearInfo,
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
                  decoration:
                      BoxDecoration(color: color, shape: BoxShape.circle),
                ),
                const SizedBox(width: 10),
                Text(stateLabel(session.state),
                    style: tt.titleSmall
                        ?.copyWith(color: color, fontWeight: FontWeight.w600)),
                const Spacer(),
                Text('ID: ${session.sessionId}',
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
            style: tt.labelMedium
                ?.copyWith(color: cs.onSurfaceVariant, letterSpacing: 0.3)),
        const SizedBox(height: 6),
        ...session.devices.map(
          (d) => _DeviceCard(device: d, appState: appState, cs: cs, tt: tt),
        ),

        const SizedBox(height: 16),

        // Actions
        Text('Nawigacja',
            style: tt.labelMedium
                ?.copyWith(color: cs.onSurfaceVariant, letterSpacing: 0.3)),
        const SizedBox(height: 8),

        _NavTile(
          icon: Icons.tune_outlined,
          label: 'Kalibracja',
          subtitle: 'Zdjęcia szachownicy i obliczenia stereo',
          enabled: session.isIdle || session.isCalibrating || session.isReady,
          onTap: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const CalibrationScreen())),
        ),
        _NavTile(
          icon: Icons.camera_outlined,
          label: 'Przechwycenie',
          subtitle: 'Synchroniczne zdjęcie pomiarowe',
          enabled: session.isReady || session.isProcessing || session.isDone,
          onTap: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const CaptureScreen())),
        ),
        _NavTile(
          icon: Icons.bar_chart_outlined,
          label: 'Wyniki',
          subtitle: 'Wymiary i raport pomiaru',
          enabled: session.isDone || appState.measurement != null,
          onTap: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const ResultsScreen())),
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
                border:
                    Border.all(color: cs.outlineVariant.withValues(alpha: 0.5)),
                borderRadius: BorderRadius.circular(8),
              ),
              padding: const EdgeInsets.all(10),
              child: appState.wsLog.isEmpty
                  ? Text('Brak zdarzeń', style: tt.bodySmall)
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: appState.wsLog.reversed
                          .take(15)
                          .map((e) => Padding(
                                padding:
                                    const EdgeInsets.symmetric(vertical: 1),
                                child: Text(e.toString(),
                                    style: tt.bodySmall?.copyWith(
                                        fontFamily: 'monospace',
                                        color: cs.onSurfaceVariant)),
                              ))
                          .toList(),
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
  final AppState appState;
  final ColorScheme cs;
  final TextTheme tt;

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
        ConnectionDot(
          active: connected,
          activeColor: AppColors.success,
          inactiveColor: cs.outline,
        ),
        const SizedBox(width: 4),
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
  final DeviceInfo device;
  final AppState appState;
  final ColorScheme cs;
  final TextTheme tt;

  const _DeviceCard({
    required this.device,
    required this.appState,
    required this.cs,
    required this.tt,
  });

  @override
  Widget build(BuildContext context) {
    // For the own device use the live local WS state; for others use server data.
    final isConnected = device.deviceId == appState.deviceId
        ? appState.wsConnected
        : device.wsConnected;

    final isOwnDevice = device.deviceId == appState.deviceId;
    final canManage = appState.isLeader && !isOwnDevice;

    final session = appState.session;
    final isLeftCam = session?.leftDeviceId == device.deviceId;
    final isRightCam = session?.rightDeviceId == device.deviceId;
    final cameraRole = isLeftCam ? 'L' : (isRightCam ? 'R' : null);

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: ListTile(
        leading: Stack(
          clipBehavior: Clip.none,
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: device.isLeader
                    ? AppColors.stateReady.withValues(alpha: 0.12)
                    : cs.surfaceContainerHighest,
                shape: BoxShape.circle,
              ),
              child: Icon(
                device.isLeader
                    ? Icons.account_circle_outlined
                    : device.isCamera
                        ? Icons.smartphone_outlined
                        : Icons.computer_outlined,
                size: 20,
                color: device.isLeader ? AppColors.stateReady : cs.onSurfaceVariant,
              ),
            ),
            if (cameraRole != null)
              Positioned(
                right: -4,
                bottom: -4,
                child: Container(
                  width: 16,
                  height: 16,
                  decoration: BoxDecoration(
                    color: isLeftCam ? cs.primary : cs.tertiary,
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                    child: Text(
                      cameraRole,
                      style: TextStyle(
                        fontSize: 9,
                        fontWeight: FontWeight.bold,
                        color: isLeftCam ? cs.onPrimary : cs.onTertiary,
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
        title: Row(
          children: [
            Flexible(child: Text(device.deviceId, style: tt.bodyMedium)),
            if (isOwnDevice) ...[
              const SizedBox(width: 6),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: cs.primaryContainer,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  'to urządzenie',
                  style: tt.labelSmall?.copyWith(color: cs.onPrimaryContainer),
                ),
              ),
            ],
          ],
        ),
        subtitle: Text(
          '${device.mac}'
          '${device.isLeader ? '  ·  lider' : ''}'
          '${!device.isCamera ? '  ·  admin' : ''}'
          '${isLeftCam ? '  ·  lewa kamera' : ''}'
          '${isRightCam ? '  ·  prawa kamera' : ''}',
          style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
        ),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            ConnectionDot(
              active: isConnected,
              activeColor: AppColors.success,
              inactiveColor: cs.outline,
            ),
            const SizedBox(width: 6),
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
            if (canManage) ...[
              const SizedBox(width: 4),
              _DeviceMenu(device: device, appState: appState, cs: cs),
            ],
          ],
        ),
      ),
    );
  }
}

class _DeviceMenu extends StatelessWidget {
  final DeviceInfo device;
  final AppState appState;
  final ColorScheme cs;

  const _DeviceMenu({
    required this.device,
    required this.appState,
    required this.cs,
  });

  Future<void> _confirmRemove(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Usuń urządzenie'),
        content: Text(
          'Usunąć ${device.deviceId} z sesji?\n'
          'Urządzenie straci połączenie i będzie musiało ponownie dołączyć.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Anuluj'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child:
                Text('Usuń', style: TextStyle(color: cs.error)),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await appState.removeDevice(device.deviceId);
    }
  }

  Future<void> _assignCamera(BuildContext context, bool asLeft) async {
    final session = appState.session;
    if (session == null) return;
    // Find the other camera device to fill the opposite role.
    final otherCams = session.devices
        .where((d) => d.isCamera && d.deviceId != device.deviceId)
        .toList();
    if (otherCams.isEmpty) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Potrzeba drugiego urządzenia z kamerą'),
        ));
      }
      return;
    }
    // If there's exactly one other camera, assign automatically.
    // Otherwise let the user see current roles and just swap.
    final otherCam = otherCams.first;
    final leftId = asLeft ? device.deviceId : otherCam.deviceId;
    final rightId = asLeft ? otherCam.deviceId : device.deviceId;
    await appState.assignCameras(leftId, rightId);
  }

  @override
  Widget build(BuildContext context) {
    final session = appState.session;
    final isLeftCam = session?.leftDeviceId == device.deviceId;
    final isRightCam = session?.rightDeviceId == device.deviceId;

    return PopupMenuButton<String>(
      icon: Icon(Icons.more_vert, size: 18, color: cs.onSurfaceVariant),
      onSelected: (action) async {
        switch (action) {
          case 'promote':
            await appState.promoteDevice(device.deviceId);
          case 'camera_on':
            await appState.toggleDeviceCamera(device.deviceId, isCamera: true);
          case 'camera_off':
            await appState.toggleDeviceCamera(device.deviceId, isCamera: false);
          case 'set_left':
            if (context.mounted) await _assignCamera(context, true);
          case 'set_right':
            if (context.mounted) await _assignCamera(context, false);
          case 'remove':
            if (context.mounted) await _confirmRemove(context);
        }
      },
      itemBuilder: (_) => [
        if (!device.isLeader)
          const PopupMenuItem(
            value: 'promote',
            child: ListTile(
              leading: Icon(Icons.star_outline),
              title: Text('Mianuj liderem'),
              contentPadding: EdgeInsets.zero,
              visualDensity: VisualDensity.compact,
            ),
          ),
        if (device.isCamera && !isLeftCam)
          const PopupMenuItem(
            value: 'set_left',
            child: ListTile(
              leading: Icon(Icons.camera_front_outlined),
              title: Text('Ustaw jako lewą kamerę'),
              contentPadding: EdgeInsets.zero,
              visualDensity: VisualDensity.compact,
            ),
          ),
        if (device.isCamera && !isRightCam)
          const PopupMenuItem(
            value: 'set_right',
            child: ListTile(
              leading: Icon(Icons.camera_rear_outlined),
              title: Text('Ustaw jako prawą kamerę'),
              contentPadding: EdgeInsets.zero,
              visualDensity: VisualDensity.compact,
            ),
          ),
        if (!device.isCamera)
          const PopupMenuItem(
            value: 'camera_on',
            child: ListTile(
              leading: Icon(Icons.smartphone_outlined),
              title: Text('Ustaw jako kamera'),
              contentPadding: EdgeInsets.zero,
              visualDensity: VisualDensity.compact,
            ),
          )
        else
          const PopupMenuItem(
            value: 'camera_off',
            child: ListTile(
              leading: Icon(Icons.computer_outlined),
              title: Text('Ustaw jako admin'),
              contentPadding: EdgeInsets.zero,
              visualDensity: VisualDensity.compact,
            ),
          ),
        PopupMenuItem(
          value: 'remove',
          child: ListTile(
            leading: Icon(Icons.person_remove_outlined, color: cs.error),
            title: Text('Usuń z sesji',
                style: TextStyle(color: cs.error)),
            contentPadding: EdgeInsets.zero,
            visualDensity: VisualDensity.compact,
          ),
        ),
      ],
    );
  }
}

class _NavTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final bool enabled;
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
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: enabled
                ? cs.primaryContainer.withValues(alpha: 0.6)
                : cs.surfaceContainerHighest,
            shape: BoxShape.circle,
          ),
          child: Icon(icon,
              size: 18, color: enabled ? cs.primary : cs.onSurfaceVariant),
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
