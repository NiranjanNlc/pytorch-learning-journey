

Actually, in deeplearning, What we do is the represent the import into some sort of number because all the processing in it are related to the number. Such representation of number are simple referred as the tensor. There are varying level of tensor:

1.  Scalar ( which is simple number nothing like 7 )
2.  Vector ( single row of number like [7 7] which represent the goals done by sabitri bhandari sambha in last 3 season .
3.  Matrix ( combination of rows and colournn , which could typically represent the goals done by national team pikayer of nepal in last 3 season )
4.  Tensor ( multiple combination of matrix in higher dimensional example goals asists and no of match played by the national team player along with their best performing video compilation and video compilation can be next tensor with field channel , color height and wooth )

```python
import torch
# lets do sme sort of tensor creation
preeti_goals_last_season = torch.tensor(7)
print(preeti_goals_last_season)
print (preeti_goals_last_season.ndim)
```

    tensor(7)
    0

Abve os just the creation of scalar lets move onto creating the vector

```python
import torch
preeti_goals_last_three_season = torch.tensor([7,6,17])
print(preeti_goals_last_three_season)
print(preeti_goals_last_three_season.ndim)
print(preeti_goals_last_three_season.shape)
```

    tensor([ 7,  6, 17])
    1
    torch.Size([3])

lets create the two dimension matrix vector which contain the stats of two player preeti and sambha

```python
import torch
player_stats = torch.tensor(
[ 
   [7,6,17], # preeti in last 3 season
   [8, 12,19 ] #sambha in last 3 season
   ]
)
print(player_stats)
print(player_stats.ndim)
print(player_stats.shape)
```

    tensor([[ 7,  6, 17],
            [ 8, 12, 19]])
    2
    torch.Size([2, 3])

Till now we have understood that deep learning is all about the manipulation and operation of the tensors. However, before the data collection, we have fucking no ideas that how the tensor looks like and what should it contains. So, we randomized its with some sort of numbers. Then these randomly genrated tensor looks at the data and adjust the tensor based on them and becomes the better representation of the patten it finds .

```python
import torch

player_skill_level = torch.rand(size = (3,3))
#player 1 defnse , shoot and tackle
#player 2 defense , shoot and tackle and so on for other player
print(player_skill_level)
print(player_skill_level.ndim)
print(player_skill_level.shape)
```

    tensor([[0.6519, 0.9088, 0.3479],
            [0.5733, 0.5850, 0.6720],
            [0.7253, 0.9354, 0.7855]])
    2
    torch.Size([3, 3])

Additionally instead of random number, we can assume their skill as zero and skill as one. as well as we can define the range for their skill level. This all defines as follows

```python
import torch
# player with the zero skill
player_skill_level = torch.zeros(size = (3,3))
print(player_skill_level)
print(player_skill_level.ndim)
print(player_skill_level.shape)
# all skill 1
player_skill_level = torch.ones(size = (3,3))
print(player_skill_level)
print(player_skill_level.ndim)
print(player_skill_level.shape)
# however torch range is deprecated so no need
```

    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])
    2
    torch.Size([3, 3])
    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])
    2
    torch.Size([3, 3])

Besides m, it is important to know about the attribute of the tensor, so we dont mess up with the things while performing operation in the tensor. Some of the attributes are shape, datatype, devices, etc

```python
import torch
player_skill_level = torch.rand(size = (3,3))
print(player_skill_level)
print(player_skill_level.ndim)
print(player_skill_level.shape)
print(player_skill_level.dtype)
print(player_skill_level.device)
```

    tensor([[0.3535, 0.6209, 0.5613],
            [0.9464, 0.6423, 0.8611],
            [0.6681, 0.0084, 0.2329]])
    2
    torch.Size([3, 3])
    torch.float32
    cpu

It is also essential to know about the data types in the tensor. Knowledge of it is basically required to let the computer know if the number 5 is an integer (like a count) or a float (like a measurement) so it knows how to handle it in calculations and how much memory to use for it. Otherwise, it would be disaster or not standard if we donot strictly define it.

For example, we should use integer data types to store the goals scored by the player and float to store average speed of the player. PyTorch has many specific datatypes (e.g., torch.float32, torch.int8), but you'll mostly work with:

*   torch.float32 - The most common type for decimals.
*   torch.int32 - A common type for integers.

```python
import torch
# Let's create tensors with different datatypes
goals = torch.tensor([5, 3, 2], dtype=torch.int32)  # Whole numbers
avg_speed = torch.tensor([8.5, 9.1, 7.8], dtype=torch.float32) # Decimals

print("Goals datatype:", goals.dtype)
print("Avg Speed datatype:", avg_speed.dtype)

# This will work: Multiplying goals by an integer (promoting to int32)
points = goals * 10
print("Points:", points, points.dtype)

# This will also work, but WARNING: It forces conversion to float!
# It's like saying "5 goals" * "10.0 points per goal" = "50.0 points"
points_float = goals * 10.0
print("Points as float:", points_float, points_float.dtype)

# This will often ERROR: Trying to add goals (int) to avg_speed (float)
# It's like adding "5 goals" + "8.5 km/h"
try:
    nonsense = goals + avg_speed
except Exception as e:
    print("ERROR!:", e)
```

    Goals datatype: torch.int32
    Avg Speed datatype: torch.float32
    Points: tensor([50, 30, 20], dtype=torch.int32) torch.int32
    Points as float: tensor([50., 30., 20.]) torch.float32
    ERROR!: ...
```
