// models.dart - wszystkie modele danych aplikacji

// ---------------------------------------------------------------------------
// Sesja
// ---------------------------------------------------------------------------

class DeviceInfo {
  final String deviceId;
  final String mac;
  final bool isLeader;
  final double joinedAt;
  final bool wsConnected;
  final int calibFrameCount;
  final int captureFrameCount;

  const DeviceInfo({
    required this.deviceId,
    required this.mac,
    required this.isLeader,
    required this.joinedAt,
    required this.wsConnected,
    required this.calibFrameCount,
    required this.captureFrameCount,
  });

  factory DeviceInfo.fromJson(Map<String, dynamic> j) => DeviceInfo(
        deviceId: j['device_id'] as String,
        mac: j['mac'] as String,
        isLeader: j['is_leader'] as bool,
        joinedAt: (j['joined_at'] as num).toDouble(),
        wsConnected: j['ws_connected'] as bool? ?? false,
        calibFrameCount: j['calib_frame_count'] as int? ?? 0,
        captureFrameCount: j['capture_frame_count'] as int? ?? 0,
      );

  static const empty = DeviceInfo(
    deviceId: '', mac: '', isLeader: false,
    joinedAt: 0, wsConnected: false,
    calibFrameCount: 0, captureFrameCount: 0,
  );
}

class SessionData {
  final String sessionId;
  final String state;
  final List<DeviceInfo> devices;
  final double createdAt;

  const SessionData({
    required this.sessionId,
    required this.state,
    required this.devices,
    required this.createdAt,
  });

  factory SessionData.fromJson(Map<String, dynamic> j) => SessionData(
        sessionId: j['session_id'] as String,
        state: j['state'] as String,
        devices: (j['devices'] as List)
            .map((d) => DeviceInfo.fromJson(d as Map<String, dynamic>))
            .toList(),
        createdAt: (j['created_at'] as num).toDouble(),
      );

  bool get isIdle => state == 'IDLE';
  bool get isCalibrating => state == 'CALIBRATING';
  bool get isReady => state == 'READY';
  bool get isProcessing => state == 'PROCESSING';
  bool get isDone => state == 'DONE';

  bool get allCaptured =>
      devices.isNotEmpty && devices.every((d) => d.captureFrameCount > 0);

  int get minCalibFrames =>
      devices.isEmpty ? 0 : devices.map((d) => d.calibFrameCount).reduce((a, b) => a < b ? a : b);
}

// ---------------------------------------------------------------------------
// Wynik pomiaru
// ---------------------------------------------------------------------------

class MeasurementResult {
  final bool validationPassed;
  final double widthMm;
  final double lengthMm;
  final double heightMm;
  final double palletRmsMm;
  final int nObjectPts;
  final int nPalletInliers;
  final List<String> issues;
  final String report;

  const MeasurementResult({
    required this.validationPassed,
    required this.widthMm,
    required this.lengthMm,
    required this.heightMm,
    required this.palletRmsMm,
    required this.nObjectPts,
    required this.nPalletInliers,
    required this.issues,
    required this.report,
  });

  factory MeasurementResult.fromJson(Map<String, dynamic> j) => MeasurementResult(
        validationPassed: j['validation_passed'] as bool,
        widthMm: (j['width_mm'] as num).toDouble(),
        lengthMm: (j['length_mm'] as num).toDouble(),
        heightMm: (j['height_mm'] as num).toDouble(),
        palletRmsMm: (j['pallet_rms_mm'] as num).toDouble(),
        nObjectPts: j['n_object_pts'] as int,
        nPalletInliers: j['n_pallet_inliers'] as int,
        issues: (j['issues'] as List).map((e) => e as String).toList(),
        report: j['report'] as String,
      );
}

// ---------------------------------------------------------------------------
// Błąd API
// ---------------------------------------------------------------------------

class ApiException implements Exception {
  final int statusCode;
  final String message;
  const ApiException(this.statusCode, this.message);

  @override
  String toString() => 'HTTP $statusCode: $message';
}
