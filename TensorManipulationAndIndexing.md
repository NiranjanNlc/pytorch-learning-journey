# Tensor Manipulation Operations

## Introduction

Till now, what we find is that we need to define the architecture of the learning model and feed the data. The model will adjust or let's say optimize the bias and weight to fit output based on the input. During this adjustment, the tensor undergoes different sorts of operations. Knowledge of those functions will definitely help us in debugging the model and defining our own model.

The goal is not to just use or drive the existing model, but to fix the one, improve the one and design the new one.

## Matrix Multiplication Example

Let's take the example of tensor matrix multiplication. Two matrices should have inner dimension same, i.e., (a,b) and (b,c) which results in a matrix of (a,c). The knowledge of the matrix inner dimension can be retrieved from the shape operation of the matrix and `matmul` is used for the multiplication of matrices.

```python
import torch

# Let's say tensor_A represents 3 samples of 2 features each.
# Shape: [3, 2] (3 rows, 2 columns)
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

# Let's say tensor_B also represents 3 samples of 2 features.
# Shape: [3, 2] (3 rows, 2 columns)
tensor_B = torch.tensor([[7, 8],
                         [9, 10],
                         [11, 12]])

# Now, let's try to matrix multiply them.
# We are trying: (3, 2) @ (3, 2)
# The inner dimensions are 2 and 3. They are NOT equal.
# This will fail.

try:
    output = torch.matmul(tensor_A, tensor_B)
except RuntimeError as e:
    print(f"Oh no! We got the classic error:\n{e}")
```

**Output:**
```
Oh no! We got the classic error:
mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)
```

Let's correct the above by transposing to make the matrix multiplication compatible.

```python
# The fix: transpose tensor_B
print("tensor_B:")
print(tensor_B)
print(f"tensor_B shape: {tensor_B.shape}\n")

print("tensor_B.T (transposed):")
print(tensor_B.T)
print(f"tensor_B.T shape: {tensor_B.T.shape}\n")

# Now let's multiply
output = torch.matmul(tensor_A, tensor_B.T)
print("Output of matrix multiplication:")
print(output)
print(f"Output shape: {output.shape}")
```

**Output:**
```
tensor_B:
tensor([[ 7,  8],
        [ 9, 10],
        [11, 12]])
tensor_B shape: torch.Size([3, 2])

tensor_B.T (transposed):
tensor([[ 7,  9, 11],
        [ 8, 10, 12]])
tensor_B.T shape: torch.Size([2, 3])

Output of matrix multiplication:
tensor([[ 23,  29,  35],
        [ 53,  67,  81],
        [ 83, 105, 127]])
Output shape: torch.Size([3, 3])
```

## Tensor Indexing

Now, the concept of indexing in the tensor. It is nothing but retrieving the element of a tensor. We will strengthen our concept of indexing with the following code.

```python
import torch

# Tensor for the stats of Preeti, Sambha and Renuka
# 1st dimension: player
# 2nd dimension: stats of player of 3 columns (goals, assists and tackles)
player_stats = torch.tensor([
    [7, 4, 34],   # Sambha
    [6, 12, 47],  # Preeti
    [0, 23, 123]  # Renuka
])

print("Full stats tensor")
print(player_stats)
print(f"Player stats shape: {player_stats.shape} -> [Players, Stats]\n")
```

**Output:**
```
Full stats tensor
tensor([[  7,   4,  34],
        [  6,  12,  47],
        [  0,  23, 123]])
Player stats shape: torch.Size([3, 3]) -> [Players, Stats]
```

Here, indexing provides important information. Let's say the newly appointed manager/coach wants to know how many goals Sambha scored. We can give him this information by indexing [0,0]:

```python
sambha_goals = player_stats[0, 0]
print(f"Sambha scored {sambha_goals} goals")
```

**Output:**
```
Sambha scored 7 goals
```

Then he just wants to know the assists:

```python
all_assists = player_stats[:, 1]
print(f"All player assists: {all_assists}")
print(f"All player assists shape: {all_assists.shape}")
```

**Output:**
```
All player assists: tensor([ 4, 12, 23])
All player assists shape: torch.Size([3])
```

He may want to know the defensive works:

```python
defensive_work = player_stats[:, 2]
print(f"All player defensive works: {defensive_work}")
print(f"All player defensive works shape: {defensive_work.shape}")
```

**Output:**
```
All player defensive works: tensor([ 34,  47, 123])
All player defensive works shape: torch.Size([3])
```

## Tensor Operations for Analysis

Not only indexing, we might want to get some sort of insight for analysis. For example, let's answer two questions:

1. What's the total contribution across all players?
2. Who had the most tackles? I want to praise them in the press conference.

```python
total_contribution = player_stats.sum(dim=0)
print(f"Total contribution across all players: {total_contribution}")
print(f"Total contribution across all players shape: {total_contribution.shape}")

most_tackle = player_stats[:, 2].argmax()
if most_tackle == 0:
    player_praised = "Sambha"
elif most_tackle == 1:
    player_praised = "Preeti"
else:
    player_praised = "Renuka"

print(f"Player praised: {player_praised}")
print(f"Index of who had the most tackles: {most_tackle}")
print(f"Who had the most tackles shape: {most_tackle.shape}")
```

**Output:**
```
Total contribution across all players: tensor([ 13,  39, 204])
Total contribution across all players shape: torch.Size([3])
Player praised: Renuka
Index of who had the most tackles: 2
Who had the most tackles shape: torch.Size([])
```

## Multi-Season Analysis

Now, let's widen our understanding by adding stats for this season too. Now, we are interested in finding out the following:

1. Total stats for each player including this season
2. Total stats for the player including this season
3. "Most Consistent Defender" award. It goes to the defender whose tackle count varied the least

```python
this_season_stats = torch.tensor([
    [7, 2, 13],
    [6, 12, 21],
    [0, 12, 32]
])

total_stats = torch.add(player_stats, this_season_stats)
print(f"Total stats for the player: {total_stats}")
print(f"Total stats for the player shape: {total_stats.shape}")

print(f"Stats for Sambha: {total_stats[0, :]}")
print(f"Stats for Sambha shape: {total_stats[0, :].shape}")

print(f"Stats for Preeti: {total_stats[1, :]}")
print(f"Stats for Preeti shape: {total_stats[1, :].shape}")

print(f"Stats for Renuka: {total_stats[2, :]}")
print(f"Stats for Renuka shape: {total_stats[2, :].shape}")

# Finding most consistent defender
two_season_tensor = torch.stack((player_stats, this_season_stats), dim=0)

print(f"Two season tensor: {two_season_tensor}")
print(f"Two season tensor shape: {two_season_tensor.shape}")

tackle_only = two_season_tensor[:, :, 2]
print(f"Tackle only: {tackle_only}")
print(f"Tackle only shape: {tackle_only.shape}")

tackle_diff = tackle_only[0, :] - tackle_only[1, :]
print(f"Tackle diff: {tackle_diff}")
print(f"Tackle diff shape: {tackle_diff.shape}")

tackle_grad = tackle_diff.argmin()
print(f"Tackle grad: {tackle_grad}")
print(f"Tackle grad shape: {tackle_grad.shape}")

if tackle_grad == 0:
    consist_defender = "Sambha"
elif tackle_grad == 1:
    consist_defender = "Preeti"
else:
    consist_defender = "Renuka"

print(f"Most Consistent Defender: {consist_defender}")

tackle_improved = tackle_diff.argmax()
if tackle_improved == 0:
    improved_defender = "Sambha"
elif tackle_improved == 1:
    improved_defender = "Preeti"
else:
    improved_defender = "Renuka"

print(f"Most Improved Defender: {improved_defender}")
```

**Output:**
```
Total stats for the player: tensor([[ 14,   6,  47],
        [ 12,  24,  68],
        [  0,  35, 155]])
Total stats for the player shape: torch.Size([3, 3])
Stats for Sambha: tensor([14,  6, 47])
Stats for Sambha shape: torch.Size([3])
Stats for Preeti: tensor([12, 24, 68])
Stats for Preeti shape: torch.Size([3])
Stats for Renuka: tensor([  0,  35, 155])
Stats for Renuka shape: torch.Size([3])
Two season tensor: tensor([[[  7,   4,  34],
         [  6,  12,  47],
         [  0,  23, 123]],

        [[  7,   2,  13],
         [  6,  12,  21],
         [  0,  12,  32]]])
Two season tensor shape: torch.Size([2, 3, 3])
Tackle only: tensor([[ 34,  47, 123],
        [ 13,  21,  32]])
Tackle only shape: torch.Size([2, 3])
Tackle diff: tensor([21, 26, 91])
Tackle diff shape: torch.Size([3])
Tackle grad: 0
Tackle grad shape: torch.Size([])
Most Consistent Defender: Sambha
Most Improved Defender: Renuka
``` 
