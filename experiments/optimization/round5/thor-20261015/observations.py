"""Summarize audited Thor phases and compute within-seed marginal effects."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent
MODES = ['certified', 'cooperative', 'predictive', 'predictive_neural']
METRICS = ['wall_sec', 'training_samples_per_wall_sec', 'p95_ms', 'p99_ms',
           'miss_rate', 'goodput', 'online_accuracy', 'final_accuracy',
           'controller_seconds', 'train_completion_seconds',
           'neural_changed_decisions', 'decisions', 'neural_log_mae', 'base_log_mae',
           'guard_overrides', 'predictive_deferrals', 'neural_updates',
           'service_requests', 'service_batches', 'neural_scored']


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary, pairs, all_rows = [], [], []
    for phase, expected in [('pilot', 3), ('main', 16), ('holdout', 4)]:
        audited = json.loads((HERE / phase / f'{phase}_summary.json').read_text())
        assert audited['complete'] and audited['runs'] == expected
        with (HERE / phase / 'results.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        all_rows.extend(dict(phase=phase, **r) for r in rows)
        if phase == 'pilot':
            continue
        audited_groups = {(r['case'], r['mode']): r for r in audited['summary']}
        for case in ['longer', 'deeper_bursty']:
            selected = [r for r in rows if r['case'] == case]
            for mode in MODES:
                group = [r for r in selected if r['mode'] == mode]
                if not group:
                    continue
                item = dict(phase=phase, case=case, mode=mode, runs=len(group))
                for metric in METRICS:
                    values = [float(r[metric]) for r in group if r[metric] != '']
                    item[metric] = mean(values) if values else None
                    item[metric + '_std'] = stdev(values) if len(values) > 1 else None
                group_audit = audited_groups[(case, mode)]
                item['mean_inference_batch'] = group_audit['mean_batch']
                item['inference_batches'] = group_audit['inference_batches']
                fractions = [100 * float(r['neural_changed_decisions']) / float(r['decisions'])
                             for r in group if r['neural_changed_decisions'] != '']
                item['neural_decision_change_pct'] = mean(fractions) if fractions else None
                summary.append(item)
            for baseline, candidate, effect in [
                ('certified', 'cooperative', 'preparation_service'),
                ('cooperative', 'predictive', 'predictive_batching'),
                ('predictive', 'predictive_neural', 'neural_residual'),
                ('cooperative', 'predictive_neural', 'combined_vs_fifo'),
            ]:
                for b in [r for r in selected if r['mode'] == baseline]:
                    match = [r for r in selected if r['mode'] == candidate and r['seed'] == b['seed']]
                    if not match:
                        continue
                    c = match[0]
                    item = dict(phase=phase, case=case, seed=int(b['seed']), effect=effect,
                                baseline=baseline, candidate=candidate)
                    for metric in ['wall_sec', 'training_samples_per_wall_sec', 'p95_ms', 'p99_ms']:
                        item[metric + '_change_pct'] = 100 * (float(c[metric]) / float(b[metric]) - 1)
                    for metric in ['miss_rate', 'online_accuracy', 'final_accuracy']:
                        item[metric + '_change_pp'] = 100 * (float(c[metric]) - float(b[metric]))
                    pairs.append(item)
    write_csv(HERE / 'summary.csv', summary)
    write_csv(HERE / 'paired_comparisons.csv', pairs)
    (HERE / 'observations.json').write_text(json.dumps(dict(
        complete=True, runs=len(all_rows), requests=sum(int(r['requests']) for r in all_rows),
        training_updates=sum(int(r['train_updates']) for r in all_rows),
        summary=summary, paired_comparisons=pairs), indent=2) + '\n')
    lines = ['| 阶段 | 设置 | 策略 | 次数 | 总耗时 s | 有效训练样本/s | P95 ms | P99 ms | 超时率 % | 在线准确率 % |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['phase']} | {r['case']} | {r['mode']} | {r['runs']} | "
                     f"{r['wall_sec']:.2f} | {r['training_samples_per_wall_sec']:.1f} | "
                     f"{r['p95_ms']:.2f} | {r['p99_ms']:.2f} | "
                     f"{100*r['miss_rate']:.3f} | {100*r['online_accuracy']:.3f} |")
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    learning = [r for r in pairs if r['effect'] == 'neural_residual']
    labels = [f"{r['phase']} / {'R20' if r['case'] == 'longer' else 'R56'} / seed {r['seed']}"
              for r in learning]
    colors = ['#0284c7' if r['case'] == 'longer' else '#d97706' for r in learning]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, layout='constrained')
    for ax, key, label in zip(axes,
            ['wall_sec_change_pct', 'p95_ms_change_pct', 'miss_rate_change_pp'],
            ['Wall time change (%)', 'Request P95 change (%)', 'Deadline misses change (pp)']):
        ax.barh(range(len(learning)), [r[key] for r in learning], color=colors, alpha=.8)
        ax.axvline(0, color='#334155', linewidth=1)
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Neural residual vs analytical predictor | Lower is better\n'
                 'Matched model and arrival traces; main and holdout shown separately')
    fig.savefig(HERE / 'learning_effects.png', dpi=170)
    fig.savefig(HERE / 'learning_effects.pdf')
    plt.close(fig)
    print('\n'.join(lines))
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()
