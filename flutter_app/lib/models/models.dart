// models.dart - wszystkie modele danych aplikacji

// ---------------------------------------------------------------------------
// Sesja
// ---------------------------------------------------------------------------

class DeviceInfo {
  final String deviceId;
  final String mac;
  final bool isLeader;
  final bool isCamera;
  final double joinedAt;
  final bool wsConnected;
  final int calibFrameCount;
  final int captureFrameCount;

  const DeviceInfo({
    required this.deviceId,
    required this.mac,
    required this.isLeader,
    this.isCamera = true,
    required this.joinedAt,
    required this.wsConnected,
    required this.calibFrameCount,
    required this.captureFrameCount,
  });

  factory DeviceInfo.fromJson(Map<String, dynamic> j) => DeviceInfo(
        deviceId: j['device_id'] as String,
        mac: j['mac'] as String,
        isLeader: j['is_leader'] as bool,
        isCamera: j['is_camera'] as bool? ?? true,
        joinedAt: (j['joined_at'] as num).toDouble(),
        wsConnected: j['ws_connected'] as bool? ?? false,
        calibFrameCount: j['calib_frame_count'] as int? ?? 0,
        captureFrameCount: j['capture_frame_count'] as int? ?? 0,
      );

  DeviceInfo copyWith({
    bool? wsConnected,
    int? calibFrameCount,
    int? captureFrameCount,
  }) =>
      DeviceInfo(
        deviceId: deviceId,
        mac: mac,
        isLeader: isLeader,
        isCamera: isCamera,
        joinedAt: joinedAt,
        wsConnected: wsConnected ?? this.wsConnected,
        calibFrameCount: calibFrameCount ?? this.calibFrameCount,
        captureFrameCount: captureFrameCount ?? this.captureFrameCount,
      );

  static const empty = DeviceInfo(
    deviceId: '', mac: '', isLeader: false, isCamera: true,
    joinedAt: 0, wsConnected: false,
    calibFrameCount: 0, captureFrameCount: 0,
  );
}

class SessionData {
  final String sessionId;
  final String state;
  final List<DeviceInfo> devices;
  final double createdAt;
  final bool hasCalibration;
  final bool hasMeasurement;
  final String? leftDeviceId;
  final String? rightDeviceId;

  const SessionData({
    required this.sessionId,
    required this.state,
    required this.devices,
    required this.createdAt,
    this.hasCalibration = false,
    this.hasMeasurement = false,
    this.leftDeviceId,
    this.rightDeviceId,
  });

  factory SessionData.fromJson(Map<String, dynamic> j) => SessionData(
        sessionId: j['session_id'] as String,
        state: j['state'] as String,
        devices: (j['devices'] as List)
            .map((d) => DeviceInfo.fromJson(d as Map<String, dynamic>))
            .toList(),
        createdAt: (j['created_at'] as num).toDouble(),
        hasCalibration: j['has_calibration'] as bool? ?? false,
        hasMeasurement: j['has_measurement'] as bool? ?? false,
        leftDeviceId: j['left_device_id'] as String?,
        rightDeviceId: j['right_device_id'] as String?,
      );

  SessionData copyWithDevices(List<DeviceInfo> devices) => SessionData(
        sessionId: sessionId,
        state: state,
        devices: devices,
        createdAt: createdAt,
        hasCalibration: hasCalibration,
        hasMeasurement: hasMeasurement,
        leftDeviceId: leftDeviceId,
        rightDeviceId: rightDeviceId,
      );

  bool get isIdle => state == 'IDLE';
  bool get isCalibrating => state == 'CALIBRATING';
  bool get isReady => state == 'READY';
  bool get isProcessing => state == 'PROCESSING';
  bool get isDone => state == 'DONE';

  bool get allCaptured {
    final cams = devices.where((d) => d.isCamera).toList();
    return cams.isNotEmpty && cams.every((d) => d.captureFrameCount > 0);
  }

  int get minCalibFrames {
    final cams = devices.where((d) => d.isCamera).toList();
    if (cams.isEmpty) return 0;
    return cams.map((d) => d.calibFrameCount).reduce((a, b) => a < b ? a : b);
  }
}

// ---------------------------------------------------------------------------
// Wynik pomiaru
// ---------------------------------------------------------------------------

class MeasurementResult {
  final bool validationPassed;
  final double widthMm;
  final double lengthMm;
  final double heightMm;
  final double volumeVoxelMm3;
  final double volumeBboxMm3;
  final double? volumeHullMm3; // null gdy convex hull niedostępny
  final double fillRatio; // voxel / bbox, "pełność" bryły [0..1]
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
    required this.volumeVoxelMm3,
    required this.volumeBboxMm3,
    required this.volumeHullMm3,
    required this.fillRatio,
    required this.palletRmsMm,
    required this.nObjectPts,
    required this.nPalletInliers,
    required this.issues,
    required this.report,
  });

  // Objętości w litrach (dm³ = mm³ / 1e6) - wygodne do wyświetlania
  double get volumeVoxelL => volumeVoxelMm3 / 1e6;
  double get volumeBboxL => volumeBboxMm3 / 1e6;
  double? get volumeHullL =>
      volumeHullMm3 == null ? null : volumeHullMm3! / 1e6;

  factory MeasurementResult.fromJson(Map<String, dynamic> j) => MeasurementResult(
        validationPassed: j['validation_passed'] as bool,
        widthMm: (j['width_mm'] as num).toDouble(),
        lengthMm: (j['length_mm'] as num).toDouble(),
        heightMm: (j['height_mm'] as num).toDouble(),
        volumeVoxelMm3: (j['volume_voxel_mm3'] as num?)?.toDouble() ?? 0,
        volumeBboxMm3: (j['volume_bbox_mm3'] as num?)?.toDouble() ?? 0,
        volumeHullMm3: (j['volume_hull_mm3'] as num?)?.toDouble(),
        fillRatio: (j['fill_ratio'] as num?)?.toDouble() ?? 0,
        palletRmsMm: (j['pallet_rms_mm'] as num).toDouble(),
        nObjectPts: j['n_object_pts'] as int,
        nPalletInliers: j['n_pallet_inliers'] as int,
        issues: (j['issues'] as List).map((e) => e as String).toList(),
        report: j['report'] as String,
      );
}

// ---------------------------------------------------------------------------
// Lokalna historia sesji (zapisywana w SharedPreferences)
// ---------------------------------------------------------------------------

/// Lekki wskaźnik do sesji, którą to urządzenie utworzyło lub do której
/// dołączyło. Pozwala wrócić do sesji (i jej danych na serwerze) po ponownym
/// uruchomieniu aplikacji albo po opuszczeniu sesji.
class SessionRef {
  final String sessionId;
  final String serverUrl;
  final bool isLeader;
  final bool isCamera;
  final double createdAt;

  const SessionRef({
    required this.sessionId,
    required this.serverUrl,
    required this.isLeader,
    this.isCamera = true,
    required this.createdAt,
  });

  Map<String, dynamic> toJson() => {
        'session_id': sessionId,
        'server_url': serverUrl,
        'is_leader': isLeader,
        'is_camera': isCamera,
        'created_at': createdAt,
      };

  factory SessionRef.fromJson(Map<String, dynamic> j) => SessionRef(
        sessionId: j['session_id'] as String,
        serverUrl: j['server_url'] as String? ?? '',
        isLeader: j['is_leader'] as bool? ?? false,
        isCamera: j['is_camera'] as bool? ?? true,
        createdAt: (j['created_at'] as num?)?.toDouble() ?? 0,
      );
}

// ---------------------------------------------------------------------------
// Informacja o pojedynczej klatce (kalibracja lub przechwycenie)
// ---------------------------------------------------------------------------

class FrameInfo {
  final int index;
  final String filename;

  const FrameInfo({required this.index, required this.filename});

  factory FrameInfo.fromJson(Map<String, dynamic> j) => FrameInfo(
        index: j['index'] as int,
        filename: j['filename'] as String,
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
