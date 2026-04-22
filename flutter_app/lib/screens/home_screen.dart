import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/app_state.dart';
import 'session_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _serverCtrl = TextEditingController(text: 'http://192.168.1.1:8000');
  final _deviceCtrl = TextEditingController();
  final _macCtrl = TextEditingController();
  final _sidCtrl = TextEditingController();
  bool _isLeader = true;
  bool _joining = false;

  @override
  void initState() {
    super.initState();
    final state = context.read<AppState>();
    _deviceCtrl.text = state.deviceId;
    _macCtrl.text = state.mac;
  }

  @override
  void dispose() {
    _serverCtrl.dispose();
    _deviceCtrl.dispose();
    _macCtrl.dispose();
    _sidCtrl.dispose();
    super.dispose();
  }

  void _applyFields(AppState state) {
    state.serverUrl = _serverCtrl.text.trim();
  }

  Future<void> _createAndJoin(AppState state) async {
    _applyFields(state);
    setState(() => _joining = true);
    final ok = await state.createAndJoin(
      _deviceCtrl.text.trim(),
      _macCtrl.text.trim(),
      true,
    );
    setState(() => _joining = false);
    if (ok && mounted) {
      Navigator.push(context, MaterialPageRoute(builder: (_) => const SessionScreen()));
    }
  }

  Future<void> _joinExisting(AppState state) async {
    final sid = _sidCtrl.text.trim();
    if (sid.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Podaj ID sesji')),
      );
      return;
    }
    _applyFields(state);
    setState(() => _joining = true);
    final ok = await state.joinExisting(
      sid,
      _deviceCtrl.text.trim(),
      _macCtrl.text.trim(),
      _isLeader,
    );
    setState(() => _joining = false);
    if (ok && mounted) {
      Navigator.push(context, MaterialPageRoute(builder: (_) => const SessionScreen()));
    }
  }

  Future<void> _syntheticTest(AppState state) async {
    _applyFields(state);
    setState(() => _joining = true);
    final result = await state.runSyntheticTest();
    setState(() => _joining = false);
    if (!mounted) return;
    if (result != null) {
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Test syntetyczny'),
          content: Text(
            'W: ${result.widthMm.toStringAsFixed(0)} mm\n'
            'L: ${result.lengthMm.toStringAsFixed(0)} mm\n'
            'H: ${result.heightMm.toStringAsFixed(0)} mm\n'
            'Walidacja: ${result.validationPassed ? "OK" : "FAIL"}',
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK')),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('StereoVision Pomiar'),
        actions: [
          IconButton(
            icon: const Icon(Icons.wifi_find),
            tooltip: 'Test połączenia',
            onPressed: _joining
                ? null
                : () async {
                    state.serverUrl = _serverCtrl.text.trim();
                    final ok = await state.testConnection();
                    if (!mounted) return;
                    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                      content: Text(ok ? 'Serwer dostępny ✓' : 'Brak połączenia z serwerem'),
                      backgroundColor: ok ? Colors.green : Colors.red,
                    ));
                  },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Error banner
            if (state.error != null)
              Card(
                color: theme.colorScheme.errorContainer,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                  child: Row(
                    children: [
                      Icon(Icons.error_outline, color: theme.colorScheme.error),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(state.error!,
                            style: TextStyle(color: theme.colorScheme.onErrorContainer)),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close),
                        onPressed: state.clearError,
                      ),
                    ],
                  ),
                ),
              ),

            // Server config card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Serwer', style: theme.textTheme.titleMedium),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _serverCtrl,
                      decoration: const InputDecoration(
                        labelText: 'URL serwera',
                        hintText: 'http://192.168.1.1:8000',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.dns),
                      ),
                      keyboardType: TextInputType.url,
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 12),

            // Device config card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Urządzenie', style: theme.textTheme.titleMedium),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _deviceCtrl,
                      decoration: const InputDecoration(
                        labelText: 'ID urządzenia',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.phone_android),
                      ),
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: _macCtrl,
                      decoration: const InputDecoration(
                        labelText: 'Adres MAC',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.router),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 12),

            // Create session card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text('Nowa sesja (lider)', style: theme.textTheme.titleMedium),
                    const SizedBox(height: 12),
                    ElevatedButton.icon(
                      onPressed: _joining ? null : () => _createAndJoin(state),
                      icon: _joining
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.add_circle),
                      label: const Text('Utwórz i dołącz jako lider'),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 12),

            // Join existing session card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text('Dołącz do sesji', style: theme.textTheme.titleMedium),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _sidCtrl,
                      decoration: const InputDecoration(
                        labelText: 'ID sesji',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.link),
                      ),
                    ),
                    const SizedBox(height: 10),
                    SwitchListTile(
                      title: const Text('Dołącz jako lider'),
                      value: _isLeader,
                      onChanged: (v) => setState(() => _isLeader = v),
                      contentPadding: EdgeInsets.zero,
                    ),
                    ElevatedButton.icon(
                      onPressed: _joining ? null : () => _joinExisting(state),
                      icon: const Icon(Icons.login),
                      label: const Text('Dołącz'),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 12),

            // Synthetic test card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text('Test syntetyczny', style: theme.textTheme.titleMedium),
                    const SizedBox(height: 4),
                    Text(
                      'Uruchamia pełny pipeline bez kamer (dane generowane syntetycznie).',
                      style: theme.textTheme.bodySmall,
                    ),
                    const SizedBox(height: 12),
                    OutlinedButton.icon(
                      onPressed: _joining ? null : () => _syntheticTest(state),
                      icon: const Icon(Icons.science),
                      label: const Text('Uruchom test syntetyczny'),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
