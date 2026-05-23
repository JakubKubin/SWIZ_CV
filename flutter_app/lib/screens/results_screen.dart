import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/app_state.dart';
import '../theme/app_theme.dart';

class ResultsScreen extends StatefulWidget {
  const ResultsScreen({super.key});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  @override
  void initState() {
    super.initState();
    final state = context.read<AppState>();
    if (state.measurement == null) {
      state.fetchMeasurementNow();
    }
  }

  void _showReport(BuildContext context, String report) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Raport pomiarowy'),
        content: SingleChildScrollView(
          child: Text(
            report,
            style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Zamknij'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Wyniki pomiaru'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_outlined),
            tooltip: 'Odśwież',
            onPressed: () => state.fetchMeasurementNow(),
          ),
        ],
      ),
      body: state.isLoading
          ? const Center(child: CircularProgressIndicator())
          : state.measurement == null
              ? _EmptyState(state: state)
              : _ResultsBody(
                  result: state.measurement!,
                  theme: theme,
                  onShowReport: () =>
                      _showReport(context, state.measurement!.report),
                ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final AppState state;

  const _EmptyState({required this.state});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;

    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.analytics_outlined,
                size: 56, color: cs.onSurfaceVariant),
            const SizedBox(height: 16),
            Text('Brak wyników',
                style: tt.titleMedium?.copyWith(color: cs.onSurfaceVariant)),
            const SizedBox(height: 8),
            Text(
              'Uruchom pomiar ze ekranu Przechwycenia\nalbo odśwież jeśli pipeline się zakończył.',
              textAlign: TextAlign.center,
              style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: () => state.fetchMeasurementNow(),
              icon: const Icon(Icons.refresh_outlined),
              label: const Text('Pobierz wyniki'),
            ),
            if (state.error != null) ...[
              const SizedBox(height: 16),
              Text(
                state.error!,
                style: tt.bodySmall?.copyWith(color: cs.error),
                textAlign: TextAlign.center,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _ResultsBody extends StatelessWidget {
  final MeasurementResult result;
  final ThemeData theme;
  final VoidCallback onShowReport;

  const _ResultsBody({
    required this.result,
    required this.theme,
    required this.onShowReport,
  });

  @override
  Widget build(BuildContext context) {
    final cs = theme.colorScheme;
    final tt = theme.textTheme;
    final passed = result.validationPassed;

    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      children: [
        // Validation badge
        Card(
          color: passed ? cs.primaryContainer : cs.errorContainer,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            child: Row(
              children: [
                Icon(
                  passed ? Icons.check_circle_outline : Icons.cancel_outlined,
                  color: passed ? cs.primary : cs.error,
                  size: 28,
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      passed ? 'Pomiar zaliczony' : 'Pomiar niezaliczony',
                      style: tt.titleSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                        color: passed
                            ? cs.onPrimaryContainer
                            : cs.onErrorContainer,
                      ),
                    ),
                    if (!passed && result.issues.isNotEmpty)
                      Text(
                        'Problemy: ${result.issues.length}',
                        style: tt.bodySmall?.copyWith(
                            color: passed
                                ? cs.onPrimaryContainer
                                : cs.onErrorContainer),
                      ),
                  ],
                ),
              ],
            ),
          ),
        ),

        const SizedBox(height: 12),

        // Dimension cards
        Text('Wymiary obiektu',
            style: tt.labelMedium
                ?.copyWith(color: cs.onSurfaceVariant, letterSpacing: 0.3)),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _DimCard(
                label: 'Szerokość',
                valueMm: result.widthMm,
                color: AppColors.dimWidth,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _DimCard(
                label: 'Długość',
                valueMm: result.lengthMm,
                color: AppColors.dimLength,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _DimCard(
                label: 'Wysokość',
                valueMm: result.heightMm,
                color: AppColors.dimHeight,
              ),
            ),
          ],
        ),

        const SizedBox(height: 16),

        // Volume estimates
        Text('Objętość obiektu',
            style: tt.labelMedium
                ?.copyWith(color: cs.onSurfaceVariant, letterSpacing: 0.3)),
        const SizedBox(height: 8),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                _MetricRow(
                  label: 'Voxel-column (height-field)',
                  value: '${result.volumeVoxelL.toStringAsFixed(2)} l',
                  ok: result.volumeVoxelMm3 > 0,
                ),
                Divider(
                    height: 20,
                    color: cs.outlineVariant.withValues(alpha: 0.5)),
                _MetricRow(
                  label: 'Bounding box (W×L×H)',
                  value: '${result.volumeBboxL.toStringAsFixed(2)} l',
                  ok: result.volumeBboxMm3 > 0,
                ),
                Divider(
                    height: 20,
                    color: cs.outlineVariant.withValues(alpha: 0.5)),
                _MetricRow(
                  label: 'Convex hull',
                  value: result.volumeHullL == null
                      ? 'niedostępna'
                      : '${result.volumeHullL!.toStringAsFixed(2)} l',
                  ok: result.volumeHullL != null,
                ),
                Divider(
                    height: 20,
                    color: cs.outlineVariant.withValues(alpha: 0.5)),
                _MetricRow(
                  label: 'Wskaźnik pełności (voxel / bbox)',
                  value: '${(result.fillRatio * 100).toStringAsFixed(0)} %',
                  ok: result.fillRatio > 0,
                ),
              ],
            ),
          ),
        ),

        const SizedBox(height: 16),

        // Quality metrics
        Text('Jakość pomiaru',
            style: tt.labelMedium
                ?.copyWith(color: cs.onSurfaceVariant, letterSpacing: 0.3)),
        const SizedBox(height: 8),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                _MetricRow(
                  label: 'RMS płaszczyzny palety',
                  value: '${result.palletRmsMm.toStringAsFixed(2)} mm',
                  ok: result.palletRmsMm < 30,
                ),
                Divider(
                    height: 20,
                    color: cs.outlineVariant.withValues(alpha: 0.5)),
                _MetricRow(
                  label: 'Punkty obiektu',
                  value: result.nObjectPts.toString(),
                  ok: result.nObjectPts > 50,
                ),
                Divider(
                    height: 20,
                    color: cs.outlineVariant.withValues(alpha: 0.5)),
                _MetricRow(
                  label: 'Inliery palety',
                  value: result.nPalletInliers.toString(),
                  ok: result.nPalletInliers > 100,
                ),
              ],
            ),
          ),
        ),

        // Issues list
        if (result.issues.isNotEmpty) ...[
          const SizedBox(height: 12),
          Text('Problemy',
              style: tt.labelMedium
                  ?.copyWith(color: cs.onSurfaceVariant, letterSpacing: 0.3)),
          const SizedBox(height: 8),
          Card(
            color: cs.errorContainer,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: result.issues
                    .map(
                      (issue) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 3),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Icon(Icons.warning_amber_outlined,
                                size: 15, color: cs.error),
                            const SizedBox(width: 6),
                            Expanded(
                              child: Text(issue,
                                  style: tt.bodySmall
                                      ?.copyWith(color: cs.onErrorContainer)),
                            ),
                          ],
                        ),
                      ),
                    )
                    .toList(),
              ),
            ),
          ),
        ],

        const SizedBox(height: 16),

        OutlinedButton.icon(
          onPressed: onShowReport,
          icon: const Icon(Icons.description_outlined),
          label: const Text('Pokaż pełny raport tekstowy'),
        ),

        const SizedBox(height: 8),
      ],
    );
  }
}

class _DimCard extends StatelessWidget {
  final String label;
  final double valueMm;
  final Color color;

  const _DimCard({
    required this.label,
    required this.valueMm,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;

    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 8),
        child: Column(
          children: [
            Text(label,
                style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
                textAlign: TextAlign.center),
            const SizedBox(height: 6),
            Text(
              valueMm.toStringAsFixed(0),
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            Text('mm',
                style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
          ],
        ),
      ),
    );
  }
}

class _MetricRow extends StatelessWidget {
  final String label;
  final String value;
  final bool ok;

  const _MetricRow({
    required this.label,
    required this.value,
    required this.ok,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;

    return Row(
      children: [
        Container(
          width: 6,
          height: 6,
          decoration: BoxDecoration(
            color: ok ? AppColors.success : AppColors.warning,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 10),
        Expanded(child: Text(label, style: tt.bodySmall)),
        Text(
          value,
          style: tt.bodySmall?.copyWith(
              fontWeight: FontWeight.w600,
              color: ok ? cs.onSurface : AppColors.warning),
        ),
      ],
    );
  }
}
