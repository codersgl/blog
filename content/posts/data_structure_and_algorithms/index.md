+++
date = '2026-03-12T22:27:38+08:00'
draft = false
title = 'Data structure and algorithms'
+++

## Base Concept of Data Structure

### Data Element

**Data element** is the basic unit of data, every data element
has some **Data Items**, which are the minimal units that consist of
**Data element**

### Data Structure

Data structure is a set of data elements which have one or more special
relationship with each other.

#### The logical structure of data

* Linear
  * Normal linear list
  * Restricted linear list
    * Stack and Queue
    * String
  * Extended linear list
    * Array

* Nonlinear
  * Set
  * Tree
    * Normal tree
    * Binary tree
  * Graph
    * Directed graph
    * Undirected graph

#### The storage structure of data

* Ordered storage
* Linked storage
* Index storage
* Hash storage

#### The operation of data

## Algorithm

Algorithm is a description of solving steps for special problems. It has five features:

1. Finiteness
2. Certainty
3. Feasibility
4. Input
5. Output

### The metric of Algorithm efficiency

1. Time Complexity

2. Spatial Complexity

## Linear list

Consecutive elements of the same type are limited in number, and every middle
element in the sequence has exactly one element immediately before it and one
element immediately after it.

### Base operation of linear list

* `InitList(&L)`: Initialize a list.
* `Length(L)`: Get the length of a list, the number of list elements.
* `LocalElem(L, e)`: Find by value.
* `GetElem(L, i)`: Find by index.
* `ListInsert(&L, i, e)`: Insert specified value e at the `i-th` element of
list L.
* `ListDelete(&L, i, &e)`: Delete the `i-th` element of list L, and return its
value to e.

### sequence list

Place the elements by using a group of consecutive memory address.

* features:
  * Easy to access and modify the element of sequence list.
  * Difficult to add and remove the element of sequence list.

In Python, Implement a sequence list by using the builtin type `list`,
For example:

```Python

a = list() # Create a empty list.
a.append(1) # Add a new element.
a.remove(1) # Remove a element.

b = [1, 2, 3] # Easier create a list.
c = b[0] # Get the first element of the list b.

```

But note to it doesn't make sure every element own same type in Python.

### Linked list

Place the elements by using some random memory address, but every element has
data and pointer that point to the address of next element.

* features:
  * Easy to add and remove the elements of link list.
  * Difficult to access and modify the elements of link list.

In Python, you need to define a node class to implement the function of link
list, like that:

```Python
class Node:
  def __init__(self, val: Any):
    self.val = val
    self.next = None

# Create a head node.
head = Node(1) # 1 -> None
head.next = Node(2) # 1 -> 2 -> None
head.next.next = Node(3) # 1 -> 2 -> 3 -> None
```

#### Doubly Linked List

Every node includes two points: one pointer points to the previous node and
another points to next node.

#### Circular Linked List

The pointer of last node points to the first node, making a circular, it could be
single direction or double direction.

#### Insert Node

1. Head Insert:
The pointer of new node points to head node, head pointer points to new node.

2. Rear Insert:
Traverse to find the last node, making it point to new node.

3. Middle Insert:
Find the target node, new node points to its next node, target node points to
new node.

#### Remove Node

1. Remove Head Node:
Head pointer points to the next node of head node, making it new head node, original
head node points to None.

2. Remove Rear Node:
Find the previous node of rear node, making it point to None.

3. Remove Middle Node:
Find the previous node of target node, making it point to the next node of target
node, then making the target node point to None.
