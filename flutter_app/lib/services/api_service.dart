// api_service.dart — klient HTTP do backendu FastAPI

import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../models/models.dart';

class ApiService {
  final String baseUrl;

  const ApiService(this.baseUrl);

  // -------------------------------------------------------------------------
  // Pomocnicy
  // -------------------------------------------------------------------------

  Future<void> _checkStatus(http.Response r) async {
    if (r.statusCode >= 400) {
      String detail;
      try {
        detail = jsonDecode(r.body)['detail'] as String? ?? r.body;
      } catch (_) {
        detail = r.body;
      }
      throw ApiException(r.statusCode, detail);
    }
  }

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  // -------------------------------------------------------------------------
  // Health
  // -------------------------------------------------------------------------

  Future<bool> healthCheck() async {
    try {
      final r = await http
          .get(_uri('/health'))
          .timeout(const Duration(seconds: 5));
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // -------------------------------------------------------------------------
  // Sesje
  // -------------------------------------------------------------------------

  Future<SessionData> createSession() async {
    final r = await http.post(_uri('/sessions'),
        headers: {'Content-Type': 'application/json'});
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<SessionData> joinSession(
    String sid,
    String deviceId,
    String mac,
    bool isLeader,
  ) async {
    final r = await http.post(
      _uri('/sessions/$sid/join'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'device_id': deviceId, 'mac': mac, 'is_leader': isLeader}),
    );
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<SessionData> getSession(String sid) async {
    final r = await http.get(_uri('/sessions/$sid'));
    await _checkStatus(r);
    return SessionData.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<void> deleteSession(String sid) async {
    final r = await http.delete(_uri('/sessions/$sid'));
    if (r.statusCode != 204 && r.statusCode != 404) await _checkStatus(r);
  }

  Future<List<SessionData>> listSessions() async {
    final r = await http.get(_uri('/sessions'));
    await _checkStatus(r);
    return (jsonDecode(r.body) as List)
        .map((j) => SessionData.fromJson(j as Map<String, dynamic>))
        .toList();
  }

  // -------------------------------------------------------------------------
  // Kalibracja
  // -------------------------------------------------------------------------

  Future<Map<String, dynamic>> uploadCalibImage(
    String sid,
    String deviceId,
    Uint8List bytes,
  ) async {
    final req = http.MultipartRequest('POST', _uri('/sessions/$sid/calibration/images'));
    req.fields['device_id'] = deviceId;
    req.files.add(http.MultipartFile.fromBytes('file', bytes, filename: 'frame.jpg'));

    final streamed = await req.send();
    final body = await streamed.stream.bytesToString();
    if (streamed.statusCode >= 400) {
      String detail;
      try {
        detail = jsonDecode(body)['detail'] as String? ?? body;
      } catch (_) {
        detail = body;
      }
      throw ApiException(streamed.statusCode, detail);
    }
    return jsonDecode(body) as Map<String, dynamic>;
  }

  Future<void> computeCalibration(String sid) async {
    final r = await http.post(_uri('/sessions/$sid/calibration/compute'));
    await _checkStatus(r);
  }

  Future<Map<String, dynamic>> getCalibrationStatus(String sid) async {
    final r = await http.get(_uri('/sessions/$sid/calibration'));
    await _checkStatus(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  // -------------------------------------------------------------------------
  // Przechwytywanie
  // -------------------------------------------------------------------------

  Future<Map<String, dynamic>> triggerCapture(String sid, int delayMs) async {
    final r = await http.post(
      _uri('/sessions/$sid/capture/trigger'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'delay_ms': delayMs}),
    );
    await _checkStatus(r);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> uploadCaptureImage(
    String sid,
    String deviceId,
    Uint8List bytes,
  ) async {
    final req = http.MultipartRequest('POST', _uri('/sessions/$sid/capture/images'));
    req.fields['device_id'] = deviceId;
    req.files.add(http.MultipartFile.fromBytes('file', bytes, filename: 'capture.jpg'));

    final streamed = await req.send();
    final body = await streamed.stream.bytesToString();
    if (streamed.statusCode >= 400) {
      String detail;
      try {
        detail = jsonDecode(body)['detail'] as String? ?? body;
      } catch (_) {
        detail = body;
      }
      throw ApiException(streamed.statusCode, detail);
    }
    return jsonDecode(body) as Map<String, dynamic>;
  }

  // -------------------------------------------------------------------------
  // Pomiar
  // -------------------------------------------------------------------------

  Future<void> runMeasurement(String sid) async {
    final r = await http.post(_uri('/sessions/$sid/measure'));
    await _checkStatus(r);
  }

  Future<MeasurementResult> getMeasurement(String sid) async {
    final r = await http.get(_uri('/sessions/$sid/measurement'));
    await _checkStatus(r);
    return MeasurementResult.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<String> getMeasurementReport(String sid) async {
    final r = await http.get(_uri('/sessions/$sid/measurement/report'));
    await _checkStatus(r);
    return r.body;
  }

  Future<MeasurementResult> syntheticMeasure() async {
    final r = await http
        .post(_uri('/measure/synthetic'))
        .timeout(const Duration(seconds: 30));
    await _checkStatus(r);
    return MeasurementResult.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }
}
