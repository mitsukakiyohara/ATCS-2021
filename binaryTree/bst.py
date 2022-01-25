"""
Mitsuka Kiyohara
ATCS 2021-2022
Binary Tree

Python program to for binary tree insertion and traversals
"""
from bst_node import Node

'''
A function that returns a string of the inorder 
traversal of a binary tree. (left --> root --> right)
Each node on the tree should be followed by a '-'.
Ex. "1-2-3-4-5-"
'''
def getInorder(root):
    result = ''
    if root: 
        #traverse left
        result += getInorder(root.left)
        #traverse root
        result += str(root.val) + '-'
        #traverse right
        result += getInorder(root.right)
    return result

'''
A function that returns a string of the postorder 
traversal of a binary tree. (left --> right --> root)
Each node on the tree should be followed by a '-'.
Ex. "1-2-3-4-5-"
'''
# A function to do postorder tree traversal
def getPostorder(root):
    result = ''
    if root: 
        #traverse left
        result += getPostorder(root.left)
        #traverse right
        result += getPostorder(root.right)
        #traverse root
        result += str(root.val) + '-'
    return result


'''
A function that returns a string of the preorder 
traversal of a binary tree. (root --> left --> right)
Each node on the tree should be followed by a '-'.
Ex. "1-2-3-4-5-"
'''
def getPreorder(root):
    result = ''
    if root:
        #traverse root
        result += str(root.val) + '-'
        #traverse left
        result += getPreorder(root.left) 
        #traverse right
        result += getPreorder(root.right) 
    return result


'''
A function that inserts a Node with the value
key in the proper position of the BST with the
provided root. The function will return the 
original root with no change if the key already
exists in the tree.
'''
def insert(root, key):
    if root is None: 
        return Node(key)
    else:
        if root.val == key: 
            return root
        elif root.val < key: 
            root.right = insert(root.right, key)
        else: 
            root.left = insert(root.left, key)
    return root

'''
Challenge: A function determines if a binary tree 
is a valid binary search tree
'''
INT_MAX = 4294967296
INT_MIN = -4294967296

# Wrapper Function
def isBST(root):
    return isBSTuntil(root, INT_MIN, INT_MAX)

def isBSTuntil(root, min, max): 
    # Empty tree = BST
    if root is None: 
        return True 
    # False if this node goes over MIN/MAX
    if root.val < min or root.val > max: 
        return False 
    # Check subtrees recusively 
    return isBSTuntil(root.left, min, root.val - 1) and isBSTuntil(root.right, root.val + 1, max)

if __name__ == '__main__':
    # Tree to help you test your code  
    root = Node(10)
    root.left = Node(5)
    root.right = Node(11)
    root.left.left = Node(3)
    root.left.right = Node(9)

    print("Preorder traversal of binary tree is")
    print(getPreorder(root))

    print("\nInorder traversal of binary tree is")
    print(getInorder(root))

    print("\nPostorder traversal of binary tree is")
    print(getPostorder(root))

    root = insert(root, 8)
    print("\nInorder traversal of binary tree with 8 inserted is")
    print(getInorder(root))

    if (isBST(root)):
        print ("Is BST")
    else:
        print ("Not a BST")