import torch

from adelt.api_call_extraction import ApiCallInfo
from .evaluator import Evaluator


def group_classes(id_to_kwdesc):
    groups = {}
    id_to_group = {}
    for lang in ['pytorch', 'keras']:
        groups[lang] = {}
        id_to_group[lang] = []
        for i, class_desc in enumerate(id_to_kwdesc[lang]):
            if class_desc[0] == ApiCallInfo.KIND_LAYER or class_desc[0] == ApiCallInfo.KIND_FUNC:
                groups[lang][class_desc[1]] = [i]
                id_to_group[lang].append(class_desc[1])
            elif class_desc[0] == ApiCallInfo.KIND_KEYWORD:
                continue
            else:
                raise NotImplementedError()
        for i, class_desc in enumerate(id_to_kwdesc[lang]):
            if class_desc[0] == ApiCallInfo.KIND_LAYER or class_desc[0] == ApiCallInfo.KIND_FUNC:
                continue
            elif class_desc[0] == ApiCallInfo.KIND_KEYWORD:
                groups[lang][class_desc[1]].append(i)
            else:
                raise NotImplementedError()
    return groups, id_to_group


def get_pair_score_greedy(p_members, k_members, id_to_kwdesc, score_mat):
    if id_to_kwdesc['pytorch'][p_members[0]][0] != id_to_kwdesc['keras'][k_members[0]][0]:
        return (float("-inf"), float("-inf")), ({}, {})
    p_result = k_result = score_mat[p_members[0], k_members[0]].item()
    local_score_mat = score_mat[p_members[1:] + [-1], :][:, k_members[1:] + [-1]]
    p_result += local_score_mat.amax(dim=1)[:-1].sum().item()
    k_result += local_score_mat.amax(dim=0)[:-1].sum().item()
    p_candidate_arr = local_score_mat.argsort(dim=1, descending=True)[:-1, :] + 1
    k_candidate_arr = local_score_mat.argsort(dim=0, descending=True)[:, :-1] + 1
    assert p_candidate_arr.size(0) == len(p_members) - 1
    assert k_candidate_arr.size(1) == len(k_members) - 1
    p_candidates, k_candidates = {}, {}
    p_candidates[p_members[0]] = [k_members[0]]
    k_candidates[k_members[0]] = [p_members[0]]
    for i in range(1, len(p_members)):
        p_candidates[p_members[i]] = [
            (k_members[j] if j < len(k_members) else -1)
            for j in p_candidate_arr[i - 1, :].tolist()
        ]
    for j in range(1, len(k_members)):
        k_candidates[k_members[j]] = [
            (p_members[i] if i < len(p_members) else -1)
            for i in k_candidate_arr[:, j - 1].tolist()
        ]
    return (p_result, k_result), (p_candidates, k_candidates)


def generate_greedy_rank(evaluator: Evaluator, s_type: str):
    score_mat = evaluator.score_mat[s_type]
    groups, id_to_group = group_classes(evaluator.id_to_kwdesc)

    # generate pairwise matches
    p_matches = [[] for _ in range(len(id_to_group['pytorch']))]
    k_matches = [[] for _ in range(len(id_to_group['keras']))]
    for p_id, p_group in enumerate(id_to_group['pytorch']):
        p_members = groups['pytorch'][p_group]
        p_matches[p_id].append((
            sum(score_mat[member, -1] for member in p_members),
            {member: [-1] for member in p_members}
        ))
    for k_id, k_group in enumerate(id_to_group['keras']):
        k_members = groups['keras'][k_group]
        k_matches[k_id].append((
            sum(score_mat[-1, member] for member in k_members),
            {member: [-1] for member in k_members}
        ))
    for p_id, p_group in enumerate(id_to_group['pytorch']):
        p_members = groups['pytorch'][p_group]
        for k_id, k_group in enumerate(id_to_group['keras']):
            k_members = groups['keras'][k_group]
            (p_api_score, k_api_score), (p_match, k_match) = get_pair_score_greedy(
                p_members, k_members,
                evaluator.id_to_kwdesc, score_mat
            )
            p_matches[p_id].append((p_api_score, p_match))
            k_matches[k_id].append((k_api_score, k_match))

    # generate candidate lists for each API keyword
    p_candidates = [[] for _ in range(len(evaluator.id_to_kwdesc['pytorch']))]
    k_candidates = [[] for _ in range(len(evaluator.id_to_kwdesc['keras']))]
    for p_id, p_group in enumerate(id_to_group['pytorch']):
        p_matches[p_id].sort(key=lambda it: it[0], reverse=True)
        for _, match in p_matches[p_id]:
            for p_member, k_member_candidates in match.items():
                p_candidates[p_member].extend(k_member_candidates)
    for k_id, k_group in enumerate(id_to_group['keras']):
        k_matches[k_id].sort(key=lambda it: it[0], reverse=True)
        for _, match in k_matches[k_id]:
            for k_member, p_member_candidates in match.items():
                k_candidates[k_member].extend(p_member_candidates)

    # from candidates to matches and a rank matrix
    p_rank = torch.full((score_mat.size(0) - 1, score_mat.size(1)), fill_value=9999, dtype=torch.long)
    k_rank = torch.full((score_mat.size(0), score_mat.size(1) - 1), fill_value=9999, dtype=torch.long)
    keyword_match = {'pytorch': [], 'keras': []}
    for p_member in range(len(evaluator.id_to_kwdesc['pytorch'])):
        cur = 0
        for k_member in p_candidates[p_member]:
            if p_rank[p_member, k_member].item() >= 9999:
                p_rank[p_member, k_member] = cur
                if cur == 0 and k_member >= 0:
                    keyword_match['pytorch'].append([p_member, k_member])
                cur += 1
    for k_member in range(len(evaluator.id_to_kwdesc['keras'])):
        cur = 0
        for p_member in k_candidates[k_member]:
            if k_rank[p_member, k_member].item() >= 9999:
                k_rank[p_member, k_member] = cur
                if cur == 0 and p_member >= 0:
                    keyword_match['keras'].append([k_member, p_member])
                cur += 1
    return keyword_match, p_rank, k_rank
