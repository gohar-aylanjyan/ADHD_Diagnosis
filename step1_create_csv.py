import pandas as pd
import numpy as np
from itertools import product
import random

print("=" * 80)
print("📊 ASRS ՏՎՅԱԼՆԵՐԻ ԲԱԶԱՅԻ ՍՏԵՂԾՈՒՄ - 1000 ՏՈՂ")
print("=" * 80)

# =============================================
# 1. ԲԵՌՆԵԼ ՀԻՆ ՏՎՅԱԼՆԵՐԸ
# =============================================

# Հին տվյալները (95 տող)
old_csv_data = """Participant Private ID,Part_A_Q1,Part_A_Q2,Part_A_Q3,Part_A_Q4,Part_A_Q5,Part_A_Q6,Part_B_Q7,Part_B_Q8,Part_B_Q9,Part_B_Q10,Part_B_Q11,Part_B_Q12,Part_B_Q13,Part_B_Q14,Part_B_Q15,Part_B_Q16,Part_B_Q17,Part_B_Q18,PartA,PartB,inattentive_sum,hyperactive_sum,total_score
9017162,3,4,4,3,4,0,1,3,1,2,4,4,4,3,2,0,0,3,18,27,21,24,45
9017163,4,4,4,4,4,1,4,4,4,4,2,4,2,3,2,3,4,2,21,38,36,23,59
9017192,1,3,2,3,3,2,3,4,3,4,3,3,3,2,2,3,4,2,14,36,27,23,50
9017217,3,3,2,3,2,3,3,4,3,3,3,3,3,2,3,2,2,2,16,33,26,23,49
9017224,2,3,2,4,3,2,4,4,3,3,2,1,1,2,4,3,3,2,16,32,28,20,48
9017244,3,4,3,3,3,1,4,4,4,4,3,3,2,2,2,2,2,3,17,35,31,21,52
9017249,2,4,1,4,3,0,1,3,3,3,3,1,3,3,0,4,1,1,14,26,22,18,40
9017255,3,4,2,2,4,3,2,2,1,2,4,4,2,4,1,4,1,3,18,30,19,29,48
9017264,3,4,4,4,4,2,4,4,4,4,4,4,4,3,2,3,4,2,21,42,35,28,63
9017273,1,4,3,4,3,1,2,4,2,3,3,2,1,2,3,2,2,2,16,28,25,19,44
9017278,1,1,1,1,3,0,1,1,2,3,2,0,2,3,2,2,2,2,7,22,13,16,29
9017279,1,2,1,1,1,0,2,3,3,3,1,2,1,3,2,1,1,1,6,23,17,12,29
9017280,1,2,3,0,4,0,1,1,4,3,1,0,0,2,2,3,0,3,10,20,15,15,30
9017282,4,4,3,3,4,3,2,4,3,3,3,2,2,4,4,3,2,4,21,36,28,29,57
9017298,1,2,3,1,3,0,1,1,1,2,1,3,1,1,1,0,1,0,10,13,13,10,23
9017301,3,4,0,4,1,0,2,4,3,3,2,2,0,0,1,0,1,0,12,18,24,6,30
9017322,2,4,2,1,4,3,3,2,0,4,3,3,4,1,2,1,2,1,16,26,20,22,42
9017328,4,3,3,4,3,2,4,4,3,3,2,3,1,2,1,3,3,3,19,32,31,20,51
9017337,4,2,2,4,1,1,4,2,4,4,2,3,3,4,4,4,2,0,14,36,28,22,50
9017338,3,3,2,1,4,2,1,2,4,3,3,0,2,3,4,3,0,4,15,29,19,25,44
9017344,3,3,4,4,4,1,3,4,3,4,3,4,2,1,4,2,1,3,19,34,29,24,53
9017346,3,3,4,3,4,4,2,4,3,4,4,4,2,3,3,4,3,3,21,39,29,31,60
9017360,1,1,2,0,3,0,2,0,1,1,3,1,1,1,2,0,1,2,7,15,9,13,22
9017371,0,1,0,0,0,0,1,2,1,0,0,0,0,2,1,1,0,0,1,8,5,4,9
9017382,3,3,1,3,3,1,1,3,3,1,1,1,3,1,3,3,1,2,14,23,19,18,37
9017383,3,3,1,1,2,0,1,2,3,2,2,2,1,0,0,0,1,1,10,15,17,8,25
9017385,2,3,2,1,1,1,1,3,1,3,2,1,0,0,1,1,1,1,10,15,17,8,25
9017390,2,2,2,2,4,0,2,2,1,4,4,2,2,1,3,3,1,2,12,27,18,21,39
9017391,1,2,2,1,2,1,0,2,2,2,2,1,3,2,1,3,1,2,9,21,13,17,30
9017392,2,2,2,1,2,0,1,1,1,2,2,2,1,3,1,1,1,3,9,19,13,15,28
9017396,3,3,2,3,3,0,2,3,1,3,2,2,3,1,0,2,1,1,14,21,21,14,35
9017401,1,1,2,2,0,0,0,1,0,1,0,1,0,1,4,0,1,0,6,9,9,6,15
9017408,1,4,3,1,1,0,3,4,3,4,1,1,1,3,4,1,1,2,10,28,24,14,38
9017412,2,2,1,3,1,0,4,4,3,1,1,1,1,2,1,1,2,2,9,23,22,10,32
9017427,1,3,1,2,4,1,2,3,2,2,2,1,2,2,3,2,2,3,12,26,18,20,38
9017429,2,2,1,2,1,0,2,2,1,1,1,1,1,1,1,1,1,1,8,14,14,8,22
9017453,4,4,4,4,2,2,4,4,4,3,3,3,2,4,3,2,4,2,20,38,35,23,58
9017454,2,3,3,4,3,2,3,3,4,3,3,2,2,2,2,3,3,2,17,32,28,21,49
9017456,2,2,3,3,4,1,3,3,3,3,3,4,2,2,2,2,3,3,15,33,25,23,48
9017466,2,4,4,2,4,2,3,4,3,4,3,4,3,4,3,3,4,3,18,41,30,29,59
9017469,4,3,0,3,2,0,3,3,3,3,1,0,2,1,4,0,1,0,12,21,23,10,33
9017470,1,3,3,0,2,0,1,2,2,2,2,3,1,4,3,2,1,1,9,24,15,18,33
9017475,4,4,4,1,2,0,3,4,3,2,3,3,4,3,4,4,1,4,15,38,26,27,53
9017477,1,3,2,1,1,0,1,4,2,2,2,2,2,1,1,2,2,1,8,22,18,12,30
9017480,2,2,2,3,3,0,1,4,1,2,1,1,3,0,0,1,1,2,12,17,18,11,29
9017481,3,3,2,4,3,0,3,4,1,4,2,1,3,0,0,0,3,0,15,21,27,9,36
9017482,3,3,3,2,4,1,2,4,3,3,3,3,1,1,1,1,3,1,16,26,26,16,42
9017484,2,2,1,4,3,0,3,0,4,4,3,4,2,4,2,0,0,1,12,27,20,19,39
9017487,1,3,4,0,2,0,1,4,2,4,3,3,3,4,4,2,0,2,10,32,19,23,42
9017488,2,3,3,1,4,2,2,3,2,3,3,3,3,2,2,1,2,1,15,27,21,21,42
9017492,1,2,1,0,1,0,0,2,1,0,3,1,0,2,1,0,0,1,5,11,7,9,16
9017496,4,4,3,1,4,3,2,4,2,3,4,4,4,4,4,3,3,2,19,39,26,32,58
9017497,1,2,1,3,4,0,1,3,3,2,3,1,2,0,0,0,3,0,11,18,19,10,29
9017498,3,2,3,2,2,1,2,2,3,2,2,1,1,1,1,3,3,3,13,24,22,15,37
9017504,2,1,1,3,0,0,1,2,1,1,0,3,1,1,1,0,1,0,7,12,13,6,19
9017506,1,2,1,3,1,0,2,1,1,2,3,3,1,1,2,1,2,1,8,20,15,13,28
9017515,1,1,2,1,0,0,2,2,1,0,1,2,1,0,2,0,1,0,5,12,11,6,17
9017516,1,1,2,2,2,0,2,1,2,3,1,2,1,2,1,2,0,2,8,19,14,13,27
9017518,3,1,2,2,1,0,1,4,2,2,1,0,0,2,0,1,1,1,9,15,18,6,24
9017523,3,3,4,4,4,1,3,4,4,4,3,2,3,3,4,1,2,3,19,36,31,24,55
9017524,3,2,1,2,2,0,1,3,2,2,2,3,0,1,0,1,2,1,10,18,18,10,28
9017532,4,2,2,4,4,0,3,4,3,2,1,2,3,3,2,1,3,2,16,29,27,18,45
9017558,3,4,0,0,2,1,0,1,0,2,1,0,3,1,0,0,1,1,10,10,11,9,20
9017559,1,1,3,1,3,0,1,1,3,1,3,1,4,1,3,3,0,2,9,23,12,20,32
9017566,3,3,3,3,3,2,3,4,3,3,3,3,2,1,3,3,0,1,17,29,25,21,46
9017570,3,3,0,2,1,1,2,4,4,2,2,0,2,0,2,0,2,0,10,20,22,8,30
9017571,3,3,4,2,4,2,2,3,4,4,4,4,4,3,4,4,4,4,18,44,29,33,62
9017589,2,3,1,3,1,0,1,2,1,2,2,1,3,2,0,1,1,3,10,19,16,13,29
9017595,3,2,3,1,1,1,1,2,1,2,1,2,2,1,3,1,1,2,11,19,16,14,30
9017596,0,3,0,3,0,0,2,3,0,1,1,0,0,0,0,0,0,0,6,7,12,1,13
9017600,1,1,0,2,0,0,2,3,1,2,0,0,2,2,1,1,1,2,4,17,13,8,21
9017602,4,4,4,0,4,2,0,4,2,3,4,4,2,2,2,0,4,0,18,27,25,20,45
9017619,3,4,3,3,3,0,3,4,3,4,2,3,2,4,4,3,1,1,16,34,28,22,50
9017620,3,3,1,1,3,0,2,3,1,2,3,3,1,0,1,1,1,2,11,20,17,14,31
9017623,0,0,0,3,0,0,1,4,0,0,0,0,0,0,0,0,0,0,3,5,8,0,8
9017636,2,0,2,3,2,0,1,1,1,0,1,0,1,0,2,1,3,1,9,12,13,8,21
9017637,2,3,3,3,3,1,3,3,2,2,3,2,2,2,2,1,2,1,15,25,23,17,40
9017641,2,4,3,3,2,0,2,4,2,3,4,3,3,2,1,0,2,1,14,27,25,16,41
9017642,2,1,2,3,3,0,1,2,0,1,2,1,2,2,2,1,1,1,11,16,13,14,27
9017657,3,3,3,3,1,0,2,3,3,2,2,3,2,3,0,0,0,1,13,21,22,12,34
9017662,2,2,2,4,2,2,3,3,1,2,3,2,3,2,3,2,2,3,14,29,21,22,43
9017681,3,2,2,2,3,3,2,3,3,2,2,3,3,3,2,3,2,3,15,31,21,25,46
9017698,2,2,3,3,2,2,2,2,1,1,3,2,3,2,3,2,3,3,14,27,19,22,41
9017709,2,4,2,2,1,1,2,2,0,1,1,2,2,1,2,1,0,1,12,15,15,12,27
9017715,2,3,3,3,3,1,3,3,3,3,2,3,2,2,2,2,4,2,15,31,27,19,46
9017722,2,1,0,2,3,0,0,1,0,0,1,0,0,0,1,0,0,1,8,4,6,6,12
9017723,2,3,3,1,4,3,3,4,2,3,4,3,3,1,3,3,2,1,16,32,23,25,48
9017730,2,3,1,1,2,1,2,1,2,2,2,1,3,1,3,1,2,1,10,21,16,15,31
9017737,3,4,3,4,4,4,4,4,4,3,4,3,4,1,3,3,3,3,22,39,32,29,61
9017738,1,1,1,2,1,2,2,1,1,2,1,1,0,1,1,2,2,2,8,16,13,11,24
9017739,3,4,1,1,2,2,3,4,2,4,4,1,4,2,4,3,3,3,13,37,25,25,50
9018354,2,1,2,1,0,0,1,2,1,4,2,1,1,2,2,0,2,1,6,19,16,9,25
9018693,3,3,2,4,3,0,4,4,3,2,2,3,2,2,1,2,3,1,15,29,28,16,44
9018708,4,4,3,4,3,3,4,4,3,4,3,3,2,3,3,4,3,3,21,39,33,27,60
9018840,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,2,1,1,2
9018901,4,3,3,3,4,0,4,3,4,3,4,3,2,4,2,3,2,1,17,35,29,23,52
9018952,3,2,3,3,3,4,2,3,2,4,2,1,2,3,4,2,3,3,18,31,25,24,49
9019035,3,3,3,2,4,0,3,3,4,3,3,0,2,3,1,2,4,2,15,30,28,17,45
9019064,1,1,0,0,1,0,1,3,2,1,1,0,0,0,0,0,0,0,3,8,9,2,11
9019102,1,2,2,1,0,0,1,1,1,0,1,1,1,0,2,1,2,1,6,12,11,7,18"""

# Ստեղծել հին DataFrame
from io import StringIO

df_old = pd.read_csv(StringIO(old_csv_data))
print(f"\n✅ Հին տվյալները բեռնված են: {len(df_old)} տող")

# =============================================
# 2. ՍՏԵՂԾԵԼ ՆՈՐ ՍԻՆԹԵՏԻԿ ՏՎՅԱԼՆԵՐ
# =============================================

print(f"\n🔄 Սինթետիկ տվյալների ստեղծում...")

# Պահպանել հին Part A և Part B զույգերը
existing_combinations = set()
for _, row in df_old.iterrows():
    part_a = tuple(row[['Part_A_Q1', 'Part_A_Q2', 'Part_A_Q3', 'Part_A_Q4', 'Part_A_Q5', 'Part_A_Q6']].values)
    part_b = tuple(row[['Part_B_Q7', 'Part_B_Q8', 'Part_B_Q9', 'Part_B_Q10', 'Part_B_Q11', 'Part_B_Q12',
                        'Part_B_Q13', 'Part_B_Q14', 'Part_B_Q15', 'Part_B_Q16', 'Part_B_Q17', 'Part_B_Q18']].values)
    existing_combinations.add((part_a, part_b))

print(f"   Հին զույգեր: {len(existing_combinations)}")


# Ֆունկցիա սինթետիկ պատասխանների ստեղծման համար
def generate_realistic_responses(num_samples, existing_combos, seed=42):
    """Ստեղծել ռեալիստիկ ASRS պատասխաններ"""
    np.random.seed(seed)
    random.seed(seed)

    new_data = []
    attempts = 0
    max_attempts = num_samples * 10

    # Տարբեր ADHD պրոֆիլներ
    profiles = [
        {'attention': 'high', 'hyperactivity': 'low', 'organization': 'medium'},  # Անուշադիր տիպ
        {'attention': 'low', 'hyperactivity': 'high', 'organization': 'medium'},  # Հիպերակտիվ տիպ
        {'attention': 'high', 'hyperactivity': 'high', 'organization': 'high'},  # Կոմբինացված տիպ
        {'attention': 'medium', 'hyperactivity': 'medium', 'organization': 'high'},  # Կազմակերպչական խնդիրներ
        {'attention': 'low', 'hyperactivity': 'low', 'organization': 'low'},  # Նորմալ
    ]

    severity_map = {
        'low': (0, 1),  # 0-1 միավոր
        'medium': (2, 3),  # 2-3 միավոր
        'high': (3, 4)  # 3-4 միավոր
    }

    while len(new_data) < num_samples and attempts < max_attempts:
        attempts += 1

        # Ընտրել պրոֆիլ
        profile = random.choice(profiles)

        # Ուշադրության հարցեր (1, 4, 7, 8, 9, 11)
        attention_range = severity_map[profile['attention']]
        part_a_responses = []
        for q in range(1, 7):
            if q in [1, 4]:  # Հիմնական ուշադրության հարցեր
                score = np.random.randint(attention_range[0], attention_range[1] + 1)
            else:
                score = np.random.randint(0, 5)
            part_a_responses.append(score)

        # Հիպերակտիվության հարցեր Part B
        hyper_range = severity_map[profile['hyperactivity']]
        org_range = severity_map[profile['organization']]

        part_b_responses = []
        for q in range(7, 19):
            if q in [5, 6, 12, 13, 14, 15, 16, 17, 18]:  # Հիպերակտիվություն
                score = np.random.randint(hyper_range[0], hyper_range[1] + 1)
            elif q in [2, 3, 10]:  # Կազմակերպվածություն
                score = np.random.randint(org_range[0], org_range[1] + 1)
            else:
                score = np.random.randint(0, 5)
            part_b_responses.append(score)

        # Ստուգել եզակիությունը
        part_a_tuple = tuple(part_a_responses)
        part_b_tuple = tuple(part_b_responses)
        combo = (part_a_tuple, part_b_tuple)

        if combo not in existing_combos:
            existing_combos.add(combo)

            # Հաշվել միավորները
            part_a_sum = sum(part_a_responses)
            part_b_sum = sum(part_b_responses)

            # Inattentive: հարցեր 1,2,3,4,7,8,9,10,11
            inattentive_indices = [0, 1, 2, 3] + [6, 7, 8, 9, 10]  # Part A (0-5) + Part B (6-17)
            all_responses = part_a_responses + part_b_responses
            inattentive_sum = sum(all_responses[i] for i in inattentive_indices if i < len(all_responses))

            # Hyperactive: հարցեր 5,6,12,13,14,15,16,17,18
            hyperactive_indices = [4, 5] + [11, 12, 13, 14, 15, 16, 17]
            hyperactive_sum = sum(all_responses[i] for i in hyperactive_indices if i < len(all_responses))

            total_score = inattentive_sum + hyperactive_sum

            new_data.append({
                'Participant Private ID': 9020000 + len(new_data),
                'Part_A_Q1': part_a_responses[0],
                'Part_A_Q2': part_a_responses[1],
                'Part_A_Q3': part_a_responses[2],
                'Part_A_Q4': part_a_responses[3],
                'Part_A_Q5': part_a_responses[4],
                'Part_A_Q6': part_a_responses[5],
                'Part_B_Q7': part_b_responses[0],
                'Part_B_Q8': part_b_responses[1],
                'Part_B_Q9': part_b_responses[2],
                'Part_B_Q10': part_b_responses[3],
                'Part_B_Q11': part_b_responses[4],
                'Part_B_Q12': part_b_responses[5],
                'Part_B_Q13': part_b_responses[6],
                'Part_B_Q14': part_b_responses[7],
                'Part_B_Q15': part_b_responses[8],
                'Part_B_Q16': part_b_responses[9],
                'Part_B_Q17': part_b_responses[10],
                'Part_B_Q18': part_b_responses[11],
                'PartA': part_a_sum,
                'PartB': part_b_sum,
                'inattentive_sum': inattentive_sum,
                'hyperactive_sum': hyperactive_sum,
                'total_score': total_score
            })

    return pd.DataFrame(new_data)


# Ստեղծել 905 նոր տող (95 հին + 905 նոր = 1000)
target_new_rows = 1000 - len(df_old)
print(f"   Նպատակ: {target_new_rows} նոր տող")

df_new = generate_realistic_responses(target_new_rows, existing_combinations)
print(f"✅ Ստեղծված է: {len(df_new)} նոր տող")

# =============================================
# 3. ՄԻԱՎՈՐԵԼ ՀԻՆ ԵՎ ՆՈՐ ՏՎՅԱԼՆԵՐԸ
# =============================================

df_combined = pd.concat([df_old, df_new], ignore_index=True)
print(f"\n✅ Ընդհանուր տվյալներ: {len(df_combined)} տող")

# Ստուգել եզակիությունը
part_a_cols = ['Part_A_Q1', 'Part_A_Q2', 'Part_A_Q3', 'Part_A_Q4', 'Part_A_Q5', 'Part_A_Q6']
part_b_cols = ['Part_B_Q7', 'Part_B_Q8', 'Part_B_Q9', 'Part_B_Q10', 'Part_B_Q11', 'Part_B_Q12',
               'Part_B_Q13', 'Part_B_Q14', 'Part_B_Q15', 'Part_B_Q16', 'Part_B_Q17', 'Part_B_Q18']

# Ստեղծել զույգերի hash
df_combined['combination_hash'] = df_combined.apply(
    lambda row: hash(tuple(row[part_a_cols].values) + tuple(row[part_b_cols].values)),
    axis=1
)

duplicates = df_combined.duplicated(subset='combination_hash', keep='first').sum()
print(f"   Կրկնակի զույգեր: {duplicates}")

if duplicates > 0:
    print(f"   ⚠️ Հեռացնում ենք {duplicates} կրկնակի տող")
    df_combined = df_combined.drop_duplicates(subset='combination_hash', keep='first')
    print(f"   ✅ Մնաց: {len(df_combined)} եզակի տող")

# Հեռացնել օժանդակ սյունակը
df_combined = df_combined.drop('combination_hash', axis=1)

# =============================================
# 4. ՎԻՃԱԿԱԳՐՈՒԹՅՈՒՆ
# =============================================

print("\n" + "=" * 80)
print("📊 ՎԻՃԱԿԱԳՐՈՒԹՅՈՒՆ")
print("=" * 80)

print(f"\nԸնդհանուր տողեր: {len(df_combined)}")
print(f"   Հին տվյալներ: {len(df_old)}")
print(f"   Նոր տվյալներ: {len(df_new)}")

print("\nՊարտ A գնահատականների բաշխում:")
print(df_combined['PartA'].describe())

print("\nՊարտ B գնահատականների բաշխում:")
print(df_combined['PartB'].describe())

print("\nՈւշադրության (Inattentive) բաշխում:")
print(df_combined['inattentive_sum'].describe())

print("\nՀիպերակտիվության (Hyperactive) բաշխում:")
print(df_combined['hyperactive_sum'].describe())


# Կատեգորիաների բաշխում
def classify_partA(score):
    if 0 <= score <= 9:
        return 'Low Negative'
    elif 10 <= score <= 13:
        return 'High Negative'
    elif 14 <= score <= 17:
        return 'Low Positive'
    elif 18 <= score <= 24:
        return 'High Positive'
    else:
        return 'Unknown'


df_combined['Category'] = df_combined['PartA'].apply(classify_partA)

print("\nԿատեգորիաների բաշխում (Part A հիման վրա):")
category_dist = df_combined['Category'].value_counts().sort_index()
for cat, count in category_dist.items():
    percentage = (count / len(df_combined)) * 100
    print(f"   {cat:20s}: {count:4d} ({percentage:5.1f}%)")

# =============================================
# 5. ՊԱՀՊԱՆԵԼ ՖԱՅԼԵՐԸ
# =============================================

print("\n" + "=" * 80)
print("💾 ՖԱՅԼԵՐԻ ՊԱՀՊԱՆՈՒՄ")
print("=" * 80)

# Պահպանել լրիվ բազան
output_file = 'ASRS_dataset_1000_rows.csv'
df_combined.drop('Category', axis=1).to_csv(output_file, index=False)
print(f"✅ Հիմնական բազա պահպանված: {output_file}")
print(f"   Չափ: {len(df_combined)} տող × {len(df_combined.columns) - 1} սյունակ")

# Պահպանել միայն հինը (backup)
backup_file = 'ASRS_original_95_rows.csv'
df_old.to_csv(backup_file, index=False)
print(f"✅ Հին բազայի backup: {backup_file}")

# Պահպանել միայն նորերը
new_file = 'ASRS_new_905_rows.csv'
df_new.to_csv(new_file, index=False)
print(f"✅ Միայն նոր տվյալներ: {new_file}")

# =============================================
# 6. ՎԱԼԻԴԱՑԻԱ
# =============================================

print("\n" + "=" * 80)
print("✅ ՎԱԼԻԴԱՑԻԱ")
print("=" * 80)

# Ստուգել սյունակները
expected_cols = ['Participant Private ID', 'Part_A_Q1', 'Part_A_Q2', 'Part_A_Q3',
                 'Part_A_Q4', 'Part_A_Q5', 'Part_A_Q6', 'Part_B_Q7', 'Part_B_Q8',
                 'Part_B_Q9', 'Part_B_Q10', 'Part_B_Q11', 'Part_B_Q12', 'Part_B_Q13',
                 'Part_B_Q14', 'Part_B_Q15', 'Part_B_Q16', 'Part_B_Q17', 'Part_B_Q18',
                 'PartA', 'PartB', 'inattentive_sum', 'hyperactive_sum', 'total_score']

print(f"✅ Բոլոր սյունակները առկա են: {set(expected_cols) == set(df_combined.columns)}")

# Ստուգել արժեքների միջակայքերը
print("\n✅ Արժեքների միջակայքերի ստուգում:")
for col in part_a_cols + part_b_cols:
    min_val = df_combined[col].min()
    max_val = df_combined[col].max()
    valid = (min_val >= 0 and max_val <= 4)
    status = "✅" if valid else "❌"
    print(f"   {status} {col}: [{min_val}, {max_val}]")

# Ստուգել Part A և Part B հաշվարկները
print("\n✅ Գումարների ստուգում:")
part_a_correct = (df_combined[part_a_cols].sum(axis=1) == df_combined['PartA']).all()
part_b_correct = (df_combined[part_b_cols].sum(axis=1) == df_combined['PartB']).all()
print(f"   Part A գումար: {'✅ Ճիշտ' if part_a_correct else '❌ Սխալ'}")
print(f"   Part B գումար: {'✅ Ճիշտ' if part_b_correct else '❌ Սխալ'}")

# Ստուգել եզակի ID-ներ
unique_ids = df_combined['Participant Private ID'].nunique()
print(f"\n✅ Եզակի ID-ներ: {unique_ids} / {len(df_combined)}")

print("\n" + "=" * 80)
print("🎉 ԱՎԱՐՏՎԱԾ Է!")
print("=" * 80)
print(f"\n📁 Ստեղծված ֆայլեր:")
print(f"   1. {output_file} - 1000 տող (հիմնական)")
print(f"   2. {backup_file} - 95 տող (հին)")
print(f"   3. {new_file} - 905 տող (նոր)")
print(f"\n💡 Հաջորդ քայլ: Օգտագործեք '{output_file}' ֆայլը Random Forest մոդելի համար")