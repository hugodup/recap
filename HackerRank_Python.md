## 1. Python: RLE Decoder

```python
import re

class RLEDecoder:
    def __init__(self, encoded_string):
        self.encoded_string = encoded_string
        self.decoded_length = 0
        self.char_counts = {}

        # Regular expression to extract character and its count
        matches = re.findall(r'([a-z])(\d+)', encoded_string)
        for char, count in matches:
            count = int(count)
            self.decoded_length += count
            self.char_counts[char] = self.char_counts.get(char, 0) + count

    def __len__(self):
        return self.decoded_length

    def count(self, char):
        return self.char_counts.get(char, 0)

# Handling input
if __name__ == "__main__":
    import sys
    input_data = sys.stdin.read().splitlines()
    
    s = input_data[0]  # Encoded string
    q = int(input_data[1])  # Number of operations
    
    decoder = RLEDecoder(s)
    
    for i in range(2, 2 + q):
        command = input_data[i].split()
        if command[0] == "len":
            print(len(decoder))
        elif command[0] == "count":
            print(decoder.count(command[1]))

```

---

## 2. Python: Non-Prines Generator

```python
import math

def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def manipulate_generator(generator, n):
    while is_prime(n + 1):
        n = generator.send(n + 1)  # Update the generator directly

def positive_integers_generator():
    n = 1
    while True:
        x = yield n
        n = x if x is not None else n + 1
        

k = int(input())
g = positive_integers_generator()
for _ in range(k):
    n = next(g)
    print(n)
    manipulate_generator(g, n)

```

---

## 3. Python Code Review: Sort Unique

```python
######Fixed Code for remove_duplicate.py:

import sys
import csv
import operator

def remove_duplicate(read_filename, write_filename, column):
    with open(read_filename, newline='') as infile:
        data = csv.reader(infile, delimiter=',')
        rows = list(data)  # Convert the iterator to a list

    newrows = []
    for row in rows:
        if row not in newrows:
            newrows.append(row)

    with open(write_filename, "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(newrows)


######Fixed Code for sort.py:

import sys
import csv
import operator

def sort_file(read_filename, write_filename, column):
    with open(read_filename, newline='') as infile:
        data = csv.reader(infile, delimiter=',')
        rows = list(data)  # Convert the iterator to a list

    sortedlist = sorted(rows, key=operator.itemgetter(int(column)))  # Use correct column

    with open(write_filename, "w", newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerows(sortedlist)


######Fixed Code for main.py:

import argparse
import sys
import sort
import remove_duplicate

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-f', '--file', help="csv file to read from", dest="input_filename", required=True)
parser.add_argument('-o', '--output', help="csv file to write to", dest="output_filename", required=True)
parser.add_argument('-c', '--column', help="column index to do action on", dest="column", required=True)
parser.add_argument('-a', '--action', help="action to run", dest="action", required=True)

args = parser.parse_args()

if args.action == "sort":
    sort.sort_file(args.input_filename, args.output_filename, int(args.column))  # Convert column to int

elif args.action == "remove_duplicate":
    remove_duplicate.remove_duplicate(args.input_filename, args.output_filename, int(args.column))  # Convert column to int

else:
    sys.exit("Error! Invalid action provided.")

```

---

## 4. Python: ATM Actions

```python
#!/bin/python3

import math
import os
import random
import re
import sys
from typing import Dict, Optional, Tuple, Any

Action = str

class State:
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return self.name

# Define states
UNAUTHORIZED = State("unauthorized")
AUTHORIZED = State("authorized")

def check_login(action_param: Optional[str], atm_password: str, atm_current_balance: int) -> Tuple[bool, int, Optional[int]]:
    return (action_param == atm_password, atm_current_balance, None)

def check_logout(action_param: Optional[str], atm_password: str, atm_current_balance: int) -> Tuple[bool, int, Optional[int]]:
    return (True, atm_current_balance, None)

def check_deposit(action_param: Optional[int], atm_password: str, atm_current_balance: int) -> Tuple[bool, int, Optional[int]]:
    return (True, atm_current_balance + action_param, None)

def check_withdraw(action_param: Optional[int], atm_password: str, atm_current_balance: int) -> Tuple[bool, int, Optional[int]]:
    if action_param <= atm_current_balance:
        return (True, atm_current_balance - action_param, None)
    return (False, atm_current_balance, None)

def check_balance(action_param: Optional[str], atm_password: str, atm_current_balance: int) -> Tuple[bool, int, Optional[int]]:
    return (True, atm_current_balance, atm_current_balance)

# Define the transition table
transition_table = {
    UNAUTHORIZED: [
        ("login", check_login, AUTHORIZED)
    ],
    AUTHORIZED: [
        ("logout", check_logout, UNAUTHORIZED),
        ("deposit", check_deposit, AUTHORIZED),
        ("withdraw", check_withdraw, AUTHORIZED),
        ("balance", check_balance, AUTHORIZED)
    ]
}

# Define the initial state
init_state = UNAUTHORIZED

if __name__ == "__main__":
    class ATM:
        def __init__(self, init_state: State, init_balance: int, password: str, transition_table: Dict):
            self.state = init_state
            self._balance = init_balance
            self._password = password
            self._transition_table = transition_table

        def next(self, action: Action, param: Optional) -> Tuple[bool, Optional[Any]]:
            try:
                for transition_action, check, next_state in self._transition_table[self.state]:
                    if action == transition_action:
                        passed, new_balance, res = check(param, self._password, self._balance)
                        if passed:
                            self._balance = new_balance
                            self.state = next_state
                            return True, res
            except KeyError:
                pass
            return False, None

    if __name__ == "__main__":
        fptr = open(os.environ['OUTPUT_PATH'], 'w')
        password = input()
        init_balance = int(input())
        atm = ATM(init_state, init_balance, password, transition_table)
        q = int(input())
        for _ in range(q):
            action_input = input().split()
            action_name = action_input[0]
            try:
                action_param = action_input[1]
                if action_name in ["deposit", "withdraw"]:
                    action_param = int(action_param)
            except IndexError:
                action_param = None
            success, res = atm.next(action_name, action_param)
            if res is not None:
                fptr.write(f"Success={success} {atm.state} {res}\n")
            else:
                fptr.write(f"Success={success} {atm.state}\n")

        fptr.close()

```

---
