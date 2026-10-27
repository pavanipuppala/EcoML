import json

with open(r'P:\ECO-ML\experiments\audit_recalculation_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== CLASSIFICATION MCDA AUDIT ===')
for prof_name, pdata in data['classification']['profile_recommendations'].items():
    print(f'\n--- Profile: {prof_name} ---')
    print('Effective Weights:', {k: round(v, 4) for k, v in pdata['effective_weights'].items()})
    print('Pareto Frontier:', pdata['pareto_frontier_models'])
    print('Top Model:', pdata['top_recommended_model'])
    for r in pdata['rankings']:
        print(f"  Rank {r['rank']}: {r['model_id']:<20} Engine Score: {r['engine_composite_score']:<7} Indep Score: {r['indep_composite_score_rounded']:<7} Diff: {r['score_diff']} Pareto: {r['engine_is_pareto']} (Indep: {r['indep_is_pareto']})")

print('\n=== REGRESSION MCDA AUDIT ===')
for prof_name, pdata in data['regression']['profile_recommendations'].items():
    print(f'\n--- Profile: {prof_name} ---')
    print('Effective Weights:', {k: round(v, 4) for k, v in pdata['effective_weights'].items()})
    print('Pareto Frontier:', pdata['pareto_frontier_models'])
    print('Top Model:', pdata['top_recommended_model'])
    for r in pdata['rankings']:
        print(f"  Rank {r['rank']}: {r['model_id']:<20} Engine Score: {r['engine_composite_score']:<7} Indep Score: {r['indep_composite_score_rounded']:<7} Diff: {r['score_diff']} Pareto: {r['engine_is_pareto']} (Indep: {r['indep_is_pareto']})")

