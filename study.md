##Diagonal Traversal

Given a matrix of M x N elements (M rows, N columns), return all elements of the matrix in diagonal order as shown in the below image.

Example:
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output:  [1,2,4,7,5,3,6,8,9]
Explanation:

Note:
The total number of elements of the given matrix will not exceed 10,000.

```javascript
var findDiagonalOrder = function(matrix) {

    if (!matrix) return [];
    if (!matrix[0]) return matrix;

    let m = matrix.length, n = matrix[0].length;
    let result = [];

    let row = 0, col = 0, d = 1;


    for (let i = 0; i < m * n; i++) {
        result.push(matrix[row][col]);

        row -= d;
        col += d;

        //bottom
        if (row >= m) {
            row = m - 1;
            col += 2;
            d = -d;
        }

        //right
        if (col >= n) {
            col = n - 1;
            row += 2;
            d = -d;
        }
        //top
        if (row < 0)  {
            row = 0;
            d = -d;
        }
        //left
        if (col < 0)  {
            col = 0;
            d = -d;
        }
    }

    return result;

};
```

There are four cases corresponding to the four sides of the matrix. We continue in the diagonal directional until we are out of bounds and then we correct.

##Construct Binary Tree from Preorder and Inorder Traversal
Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7

```javascript
var buildTree = function(preorder, inorder) {

    if(!preorder.length) return null;

    let rootVal = preorder[0];
    let rootInd = inorder.indexOf(rootVal);

    let root = new TreeNode(rootVal);

    root.left = buildTree(preorder.slice(1, rootInd + 1), inorder.slice(0, rootInd));
    root.right = buildTree(preorder.slice(rootInd + 1), inorder.slice(rootInd + 1));

    return root;


};
```

remember to draw out balanced tree to help remember the indexes. Buildtree is called n times - once for each node.

The total runtime thus depends on the number of recursion levels. If you have a approximately balanced tree, the depth is O(log n), thus we get to O(n · log n) at all. As the only necessarily slow part is the search of the root node in the inorder array, I guess we could optimize that even a bit more if we know more about the tree.

In the worst case we have one recursion level for each node in the tree, coming to complexity O(n·n).

##Valid Parentheses
```javascript
var isValid = function(s) {

    let stack = [];

    let open  = {
        '{': '}',
        '[': ']',
        '(': ')',
    }

    let closed  = new Set(['}', ']', ')']);



    for (let i = 0; i < s.length; i++) {

        if (open[s[i]]) {
            stack.push(s[i]);
        } else if (closed.has(s[i])) {
            let match = stack.pop();
            if (open[match] !== s[i]) return false;
        }
    }


    return stack.length ? false : true;
};
```
##Write a function that takes a string and returns the longest substring without repeating characters.
```javascript
var lengthOfLongestSubstring = function(s) {

    let current = new Set();
    let length = 0;
    let head = 0;
    let tail = 0;

    while (tail < s.length) {
        if (!current.has(s[tail])) {
            current.add(s[tail]);
            tail++;
            length = Math.max(current.size, length);
        } else {
            current.delete(s[head])
            head++;
        }

    }

    return length;


};
```

##Fibonacci (recursive and non-recursive)
```javascript
function fib(n, memo = {}) {

  if (n <= 1) return n;

  let fibMinusOne = memo[n - 1] || fib(n - 1, memo);
  let fibMinusTwo = memo[n - 2] || fib(n - 2, memo);
  let ans = fibMinusOne + fibMinusTwo;

  memo[n] = ans;

  return ans;
}

function bottomUpFib(n) {
  if (n <= 1) return n;

  let hold = [0, 1];

  for (let i = 0; i < (n - 1) / 2; i++) {
    hold[0] = hold[0] + hold[1];
    hold[1] = hold[0] + hold[1];
  }

  return n % 2 === 0 ? hold[0] : hold[1]
}
```

##level order traversal

```javascript
var levelOrder = function(root) {
    let levelOrd = [];
    let q = [root];
    let current;

    while (q.length) {
        let size = q.length;

        let level = [];
        for (let i = 0; i < size; i++) {
            current = q.shift();
            if (current.left) q.push(current.left);
            if (current.right) q.push(current.right);
            level.push(current.val)
        }
        levelOrd.push(level);
    }

    return levelOrd;
};

```

##api communications
```javascript
router.get('/users/:userId/books/:bookId', function (req, res) {
  res.json(req.params)
})
```

req.query
app.use to plug in routers

##three sum

```javascript
var threeSum = function(nums) {

   let triplets = [];
  nums = nums.sort((a, b) => a - b);

  for(let i = 0; i < nums.length-2; i++) {
    if(i == 0 || nums[i] > nums[i - 1]) {
      let lo = i + 1;
      let high = nums.length - 1;

      while(lo < high) {
        let sum = nums[i] + nums[lo] + nums[high];
        if(sum == 0) {
          triplets.push([nums[i], nums[lo], nums[high]]);

          //important to remember and easy to forget
          lo++;
          high--;
          //skip duplicates from lo
          while(lo<high && nums[lo]==nums[lo-1])
            lo++;

          //skip duplicates from high
          while(lo<high && nums[high]==nums[high+1])
            high--;
        } else if(sum < 0) {
          lo++;
        } else {
          high--;
        }
      }
    }
  }

  return triplets;
};
```

n^2 time

##merge two sorted lists

In Javascript, you can use the sort() function but remember that its default sort method is based on unicode comparisons.

```javascript
function mergeTwoLists(l1, l2){
		if(l1 == null) return l2;
		if(l2 == null) return l1;
		if(l1.val < l2.val){
			l1.next = mergeTwoLists(l1.next, l2);
			return l1;
		} else{
			l2.next = mergeTwoLists(l1, l2.next);
			return l2;
		}
}
```
##qiucksort
```javascript
function quickSort(nums, lo = 0, hi = arr.length - 1) {

  if (lo >= hi) return;

  for (var i = lo, j = lo; j < hi; j++) {
        if (nums[j] <= nums[hi]) {
            swap(nums, i++, j);
        }
    }
    swap(nums, i, j);


    quickSort(nums, i + 1, hi);
    quickSort(nums, lo, i - 1);
};

function swap(nums, i, j) {
    [nums[i], nums[j]] = [nums[j], nums[i]]
}
```

time and space of sorting

Mergesort is not dependant upon the array so its time is always nlogn. we also need n space for the extra arrays.

quicksort is often used in real worlds applications because it can reliably be implemnetd in nlogn time and can n2 time can be avoided. It is also convenient  becuase of the very low space cost: either constant space or logn for recursive calls

##bfs and dfs
```javascript
function TreeNode(val) {
  this.val = val;
  this.left = this.left = null;
}

function bfs(root) {
  let q = [root];

  while (q.length) {
    let size = q.length;

    for (i = 0; i < q.length; i++) {
      let curr = q.shift();
      console.log(curr.val) //processing
      if (curr.left) q.push(curr.left);
      if (curr.right) q.push(curr.right);
    }
  }

}

function dfs(root) {
  if (root.left) dfs(root.left);
  console.log(root.val)
  if (root.right) dfs(root.right);
}

function dfsGraph(root) {
  if (!root) return;
  console.log(root.val)
  root.visited = true;
  for (let child of root.chilren) {
    if (!child.visited) {
      dfs(child);
    }
  }
}

function bfsGraph(root) {
  let q = [root];

  while (q.length) {
    let curr = q.shift();
    console.log(curr);
    root.visited = true;

    for (let node in curr.children) {
      if (!node.visited) {
        node.visited = true;
        q.push(node)
      }
    }
  }
}
```

bfs space is n/2 - 1 or n.

##mergesort

```javascript
function mergeSort(arr) {
  if (arr.length === 1) return arr;

  let midPoint = Math.floor(arr.length/2);

  let leftSorted = mergeSort(arr.slice(0, midPoint));
  let rightSorted = mergeSort(arr.slice(midPoint, arr.length));

  return merge(leftSorted, rightSorted);
}

function merge(left, right) {
  let result = [];

  while (left.length && right.length){
    if (left[0] < right[0]) {
      result.push(left.shift());
    } else {
      result.push(right.shift());
    }
  }

  return left.length > 0 ? result.concat(left) : result.concat(right);
}
```
lowest common ancestor
```javascript
var lowestCommonAncestor = function(root, p, q) {

    if (!root) return null;

    if (root.val === p.val || root.val === q.val) return root;

    let left = lowestCommonAncestor(root.left, p, q);
    let right = lowestCommonAncestor(root.right, p, q);

    if (left && right) return root;
    if (left) return left;
    if (right) return right;

};
```

##How to create objects in Javascript
new keyword or objet.create

Object.create(proto[, propertiesObject])

Very simply said, new X is Object.create(X.prototype) with additionally running the constructor function. (And giving the constructor the chance to return the actual object that should be the result of the expression instead of this.)

##Databases

#Denormalized vs. Normalized:
Normalized db’s are designed to minimize redundancy while denormailzed dbs are designed to optimize read time. Denormilzation is commonly used to create highly scalable systems. In a normalized db courses might have a foreign key for teachers (with no redundancy) while in a denormalized db we might store the teachers name in the courses table.

#Acid

Transactions are single logical units of work (read and write). ACID principles are followed to maintain consistency.

A - Atomicity (all or nothing)
The entire transaction takes place or nothing happens at all. Therefor, a transaction can be aborted, and changes made to the db are not visible, or a transaction can be commited in which case they are visible.

C - concistency - Consistency ensures that a transaction can only bring the database from one valid state to another, maintaining database invariants: any data written to the database must be valid according to all defined rules, including constraints, cascades, triggers, and any combination thereof. This prevents database corruption by an illegal transaction, but does not guarantee that a transaction is correct.

I - Isolation − In a database system where more than one transaction are being executed simultaneously and in parallel, the property of isolation states that all the transactions will be carried out and executed as if it is the only transaction in the system. No transaction will affect the existence of any other transaction.

D - Durability - Once transactions are committed they must be written to a persistent memory and be able to survive system failure.

##Systems design

#load balancing

#cacheing

##node
#•What makes Node different than traditional, non-blocking web servers? How do you scale a Node application?
Node is really fast. Its non-blocking which means its really good at reading and writing and these operations don't block the stack. The core v8 javascript engine is also very fast. It allows for server-side rendering which can increase key metrics like time to render and time to interact.

The native Node.js cluster module is the basic way to scale a Node app on a single machine. A single instance of Node.js runs in a single thread. To take advantage of multi-core systems, the user will sometimes want to launch a cluster of Node.js processes to handle the load.

##error first callback (convention in node)
As a convention, callbacks in Node take the error object as the first argument, which may be nice becuase if you have an error you want to handle it first.

##last cb
in functions like fs.readfile the callback is always last.

##streams
Uses events to pass data, communicate errors, and the end of an input. Data is passed by using buffer objects or strings. Streams are readable, writable, both and transformational.

##clusters
Node is single threaded, so in order to take advantage of all of the cpu's of a machine we use clusters. If we didnt use clusters and we were doing a lot of processing, the other cpu's would be mostly dormant. The process object will tell you how many cores and you usually see a for loop forking once for every core.

The master forks the child processes. Each child process calls listen() but the master opens the socket and sends the requests to the children (round robin) who are listening on the port. Events coordinate the master and children: fork, online, listening, disconnect, exit, and setup.

```javascript
if (cluster.isMaster) {
  console.log(`Master ${process.pid} is running`);

  // Fork workers.
  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  cluster.on('exit', (worker, code, signal) => {
    console.log(`worker ${worker.process.pid} died`);
  });
} else {
  // Workers can share any TCP connection
  // In this case it is an HTTP server
  http.createServer((req, res) => {
    res.writeHead(200);
    res.end('hello world\n');
  }).listen(8000);
```

GENERAL QUESTIONS

##Subtree of Another Tree
Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

```javascript
var isSubtree = function(s, t) {

    let sPre = preorder(s);
    let tPre = preorder(t);

    return sPre.indexOf(tPre) !== -1;

};

function preorder(root, str = [""]) {
    if (!root) {
        str[0] += '*';
        return;
    }
    str[0] += " " + root.val + " "; //white space is important! Think about root 12** and root 2**
    preorder(root.left, str);
    preorder(root.right, str);
    return str[0];
}
```

In this approach we do a preorder traversal of both trees and convert their traversals into easily comparable strings. Preorder is unique when we include the null nodes as special charachters. Str is put inside an array so that we are passing the actual traversal. If the str were just a string then we would only return the root.

Alternatively we call isSameTree on every node.

```javascript
var isSubtree = function(s, t) {

    if (!s) {
        return false;
    } else if (s.val === t.val && isSameTree(s, t)) { //only calling sametree on equal vals
        return true;
    }

    return isSubtree(s.left, t) || isSubtree(s.right, t)

};

function isSameTree(s, t) {
    if (!s && !t) return true;
    if (!s || !t) return false;

    return s.val === t.val && isSameTree(s.left, t.left) && isSameTree(s.right, t.right);
}
```

The naive runtime is O(s * t) but we can see that we only call isSameTree once for every matching value. The runtime is closer to O(n + km). where k is the number of occurences of t's root in s.

### inorder successor
```javascript
var inorderSuccessor = function(root, p) {


    if (root === null) return null;

    if (root.val <= p.val) {
        return inorderSuccessor(root.right, p);
    } else {
        let left = inorderSuccessor(root.left, p);
        return left ? left : root;
  }
};
```

Just search from root to bottom, trying to find the smallest node larger than p and return the last one that was larger.

## sort colors

Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note: You are not suppose to use the library's sort function for this problem.
```javascript
var sortColors = function(nums) {
    // 1-pass
    let p1 = 0,
        p2 = nums.length - 1,
        index = 0;
    while (index <= p2) {
        if (nums[index] == 0) {
            nums[index] = nums[p1];
            nums[p1] = 0;
            p1++;
        }
        if (nums[index] == 2) {
            nums[index] = nums[p2];
            nums[p2] = 2;
            p2--;
            index--;
        }
        index++;
    }
};
```

use count sort for 2 pass solution. otherwise this can be done in one pass using some techniques of quick sort.

count sort

```javascript
unction countingSort(theArray, maxValue) {

    // array of 0's at indices 0...maxValue
    var numCounts = [];
    for (var i = 0; i < maxValue + 1; i++) {
        numCounts[i] = 0;
    }

    // populate numCounts
    theArray.forEach(function(num) {
        numCounts[num] += 1;
    });

    // populate the final sorted array
    var sortedArray = [];
    var currentSortedIndex = 0;

    // for each num in numCounts
    for (var num = 0; num < numCounts.length; num++) {
        var count = numCounts[num];

        // for the number of times the item occurs
        for (var i = 0; i < count; i++) {

            // add it to the sorted array
            sortedArray[currentSortedIndex] = num;
            currentSortedIndex++;
        }
    }

    return sortedArray;
```

getRandom BST

```javascript
function TreeNode(val) {
  this.val = val;
  this.left = this.right = null;
  this.size = 1;
}

TreeNode.prototype.getRandom = function() {
  let random = Math.floor(Math.random() * this.size);
  let leftSize = this.left ? this.left.size : 0;
  if (random < leftSize) {
    return this.left.getRandom();
  } else if (random === leftSize){
    return this;
  } else {
    return this.right.getRandom();
  }

}

TreeNode.prototype.insert = function(val) {

  this.size++;

  if (this.val < val) {
    if (!this.right) {
      this.right = new TreeNode(val);
    } else {
      this.right.insert(val);
    }
  } else {
    if (!this.left) {
      this.left = new TreeNode(val);
    } else {
      this.left.insert(val);
    }
  }
}

TreeNode.prototype.find = function(val) {
  if (this.val === val) {
    return this;
  } else if (this.val < val) {
    return this.right ? this.right.find(val) : null;
  } else {
    return this.left ? this.left.find(val) : null;
  }
}
```

the naive solution would be to build a traversal of the tree and get a random number which represents the index in the traversal. However we can do better. Each node has a size which represents the size of the tree. If there are more nodes on the left we should go left more than we should go right. if there are 10 nodes on the left and 5 on the right, then 10/16 times should go left while 5/16 we should right. 1/16 times we should return the root.

The base case is when we call getRandom on a leaf node where random is 0 becuase size is 1 and leftSize is 0 so we return this.

##Find Peak Element
A peak element is an element that is greater than its neighbors.

Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that nums[-1] = nums[n] = -∞.

```javascript
var findPeakElement = function(nums, low = 0, high = nums.length - 1) {
    if(low == high)
        return low;
    else {
        let mid1 = Math.floor((low+high)/2);
        let mid2 = mid1+1;
        if (nums[mid1] > nums[mid2]) {
            return findPeakElement(nums, low, mid1);
        } else {
            return findPeakElement(nums, mid2, high);
        }
    }
};
```

We use a binary search to find the peak. If the right value is less than the left we find the peak on the left side and vice-versa. We use binary search to find the local max in logn time.



###search in 2d matrix

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
Example:

Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.

Given target = 20, return false.

```javascript
var searchMatrix = function(matrix, target) {

    if (!matrix.length) return false;

    let row = 0;
    let col = matrix[0].length - 1;

    while (row < matrix.length && col >= 0) {
        const current = matrix[row][col];
        if (current === target) {
            return true;
        } else if (current > target) {
            col--;
        } else {
            row++;
        }
    }

    return false;

};
```

We start at the top right corner (or bottom left corner). If the current is greater than the target we need to decrease the current and so we move to the left (col--). if the current is less than the target we move down with row++. the runtime is n + m.

##search rotated

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

```javascript
var search = function(nums, target, lo = 0, hi = nums.length - 1) {

    let mid = Math.floor((lo + hi) / 2);

   // console.log(lo, hi, mid)

    if (nums[mid] === target) {
        return mid;
    }

    if (lo > hi) return -1;

    //if left is sorted
    if (nums[lo] < nums[mid]) {
        if (target >= nums[lo] && target < nums[mid]) {
            //search left
            return search(nums, target, lo, mid -1)
        } else {
            //search right
            return search(nums, target, mid + 1, hi)
        }
    } else if (nums[mid] < nums[lo]){
        if (target > nums[mid] && target <= nums[hi]) {
           //search right
            return search(nums, target, mid + 1, hi)
        } else {
            return search(nums, target, lo, mid -1)
        }
    } else if (nums[lo] === nums[mid]){ //left and right half are all repeats
        if (nums[mid] !== nums[hi]) {  //if right is different search it
            return search(nums, target, mid + 1, hi)
        } else { //we have to search both halves
          let result = return search(nums, target, lo, mid -1);
          if (result === -1) {
            return search(nums, target, mid + 1, hi)
          } else {
            return result;
          }
        }
    }
};
```

if left is sorted than we check if target is in range and search there. else search right and vice versa. else there are duplicates as in [2, 2, 2, 3, 4, 2] and we need to search right and left.

```javascript
var serialize = function(root) {
    let seri = "";

    function dfs(root) {
        if (!root) {
            seri += ' * ';
            return;
        }
        seri += " " + root.val + " ";
        dfs(root.left);
        dfs(root.right);
    }
    dfs(root);

    return seri;
};

/**
 * Decodes your encoded data to tree.
 *
 * @param {string} data
 * @return {TreeNode}
 */

var deserialize = function(data) {
    data = data.split(" ").filter((elem) => elem !== "");

    function dfs(data) {

        if (data[0] === "*") {
            return null;
        }
        let root = new TreeNode(+data[0]);
        data.shift();
        root.left = dfs(data);
        data.shift();
        root.right = dfs(data);

        return root;

    }

    return dfs(data);

};
```

```javascript
function peaksAndValleys(arr) {
  for (let i = 1; i < arr.length; i+=2) {
    let maxIndex = getMaxInd(arr, i);
    if (maxIndex !== i) {
      [arr[maxIndex], arr[i]] = [arr[i], arr[maxIndex]];
    }
  }
}

function getMaxInd(arr, i) {
  if (arr[i] > arr[i + 1] && arr[i] > arr[i - 1]) {return i}
  if (arr[i + 1] > arr[i] && arr[i + 1] > arr[i - 1]) {return i + 1};
  if (arr[i - 1] > arr[i] && arr[i - 1] > arr[i + 1]) {return i - 1 }

  return i;
}
```

Triangular series:
A triangular series always starts with 1 and increases by 1 with each number.

n(n + 1) / 2 or (n^2 + n) / 2

##What do the terms “CPU bound” and “I/O bound” mean?
A program is CPU bound if it would go faster if the CPU were faster, i.e. it spends the majority of its time simply using the CPU (doing calculations). A program that computes new digits of π will typically be CPU-bound, it's just crunching numbers.

A program is I/O bound if it would gfaster if the I/O subsystem was faster. Which exact I/O system is meant can vary; I typically associate it with disk, but of course networking or communication in general is common too. A program that looks through a huge file for some data might become I/O bound, since the bottleneck is then the reading of the data from disk (actually, this example is perhaps kind of old-fashioned these days with hundreds of MB/s coming in from SSDs).

##tcp sockets



##web application architecture

##authentication

Authentication refers to the process of checking that an entity is who they claim to be, i.e. a login process. This is different from authorization, which is more about controlling what resources a client should and should not have access to.

Local authentication as a term is contrasted with foreign strategies (or provider strategies) that come from third-party applications like Facebook and Twitter. Simply put, a user can decide to set up an account "directly", by providing a username and password, or "indirectly", through some other web application that both of you trust.

Sessions in Node
Sessions are ways of keeping a user logged in. One we can do this is by using cookies. Cookies allow you to store a user’s information inside a file on their browser. The browser then sends that info back on every request instead of sending the username and password on every request.

You can set cookies in responses. For instance, the “Set-Cookie” header might set the cookie value to a string like “session=r@rdegges.com”. Just like that, the user’s browser will store and pass along a cookie the next time they visit your site. Your server can grab the email and pull their profile information from your database if needed.

Cookies have a duration, after which they are invalidated (or logged out) but they can be extended by setting an active duration which will increase the time based on interaction with the site. Client-session libraries will also ask for a secret which should be a high-entropy string that you choose to encrypt the cookies.

##•What is a JS closure?
A closure is the combination of a function bundled together (enclosed) with references to its surrounding state (the lexical environment). In other words, a closure gives you access to an outer function’s scope from an inner function. In JavaScript, closures are created every time a function is created, at function creation time.
Examples: callbacks, data privacy, partial application

##Load Balancing

Load balancers are used to increase application availability and responsiveness while also preventing any one application server from becoming a single point of failure. It does this by spreading a load of traffic amongst a cluster of servers. Additionally, if a server is not responding or is responding with a high rate of error, LB will stop sending requests to that server.

Web server: serves content to the web using http protocol.
Application server: hosts and exposes business logic and processes. (not limited to http)
Database server: hosts the database
Load balancers can go in between all of these servers. Load balancer can be a single point of failure, to overcome this a second load balancer can be connected to the first to form a cluster. Each LB monitors the health of the other and since both of them are equally capable of serving traffic and failure detection, in the event the main load balancer fails, the second load balancer takes over.

##Cacheing
Load balancing helps you scale horizontally across an ever-increasing number of servers, but caching will enable you to make vastly better use of the resources you already have, as well as making otherwise unattainable product requirements feasible.
Caches can exist at all levels of architecture but are normally placed near the front-end as to not tax downstream levels.
Cahces can exist in memory, on disk, or CDN. CDN’s are good for serving up large amounts of static media.

Invalidation:
 If the data is modified in the database, it should be invalidated in the cache, if not, this can cause inconsistent application behavior.
Write-through: Write to both cache and db (high latency for double write)
Write around: write to db (cache misses on writes)
Write back: write to cache (low latency, poential for data loss)

##single source of truth (redux)
In redux your state exists in a single object that can be accessed through the global store. This makes code easier to debug and introspect.

##What are “actions” in Redux?
Actions represent events in the lifecycle of an application. Every action is an object with a type and payload of data. They are the means of sending data from an application to the store.

##What is the role of reducers in Redux?
Reducers take in actions and return a new state. It is important that they are pure functions and dont have sideffects. They can be combined in larger applications.

##What is ‘Store’ in Redux?
The store is the holder of the application state which also provides other methods like dispatch actions and registering listeners.

How to talk about life socks

How to persist the cart on refresh:
For non-logged in users we persisted the data to localStorage.

for signed in users we persisted the cart in the database. This allowed us to trace a cart to a user, and would allow users to log in from anywhere and still view their pending cart.

Loading items from localStorage or db.

What happens to localStorage if the browser closes?

we have window.localStorage (stores data with no expiration) and window.sessionsStorage (stores data when for one session until the browser tab is closed)

##sessions

req.session.id = user.id

```javascript
app.post('/register', (req, res) => {
  //hash password so its not plain text
  req.body.password = bcrypt.hashSync(req.body.password, 14);
  //create a new user object
  const user = _.pick(req.body, 'name', 'email', 'password');
  User.create(user)
  .then((user) => {
    req.session.userId = user.id;
    res.redirect("/dashboard")
  })
  .catch((err) => {
    console.error(err);
  })

})

app.post('login', (req, res) => {
  User.findOne({
    where: {
      email: req.body.email
    }
  })
  .then((user) => {
    if (!user) {
      res.status(401).send('Wrong username and/or password')
    } else if (!bcrypt.compareSync(user.password, req.body.password)) {
      res.render("Login", {error: "Wrong password"})
    } else {
      //use session object
      req.session.userId = user.id;
      res.redirect("/dashboard");
    }
  })
  .catch((err) => {

  })
})

app.get("dashboard", (req, res) => {
  if (!(req.sesssion && req.session.userId)) {
    return res.redirect("/login")
  }

  User.findOne({
    where:
      id: req.session.userId
  })
  .then((user) => {
    if (!user) {
      return res.redirect("/login");
    }
  })
  .catch((err) => {
    console.error(err)
  })
})

router.post('/logout', (req, res) => {
  req.logout();
  req.session.destroy();
  res.redirect("/dashboard")
})
```

replace('/\W/', "")


