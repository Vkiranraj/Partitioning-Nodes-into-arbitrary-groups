#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from itertools import combinations 
import functools
import heapq
import queue
import copy
import os
from os import path
import math


# In[2]:


#creating a comparable object

@functools.total_ordering
class ComparableCombinations:
    def __init__(self, cluster, best,size, operation ):
        self.cluster = cluster
        self.best = -best
        self.size_after_combination = size
        self.type = operation
  

    def __gt__(self, other):
        return self.best > other.best

    def __eq__(self, other):
        return self.best == other.best

    def __repr__(self):
        return repr(self.cluster)
    
    def getitem(self):
        return self.cluster
    


# In[ ]:



#creating student class, each node in the graph represents a student
class Student:

    def __init__(self, name, stress_matrix, happiness_matrix, room = None):
        self.name = name
        self.room = room
        self.stress_contributed = np.expand_dims(np.array(stress_matrix)[name], 0)
        self.happiness_contributed = np.expand_dims(np.array(happiness_matrix)[name], 0)
        self.pos_in_stress_and_happy_matrix = 0
        self.stress_happiness_ratio = 0 

    def my_stress(self):
        return (self.stress_contributed @ self.room.mycontent())/2
        
        
    def my_happiness(self):
        return (self.happiness_contributed @ self.room.mycontent())/2
     

class Room:
    def __init__(self, name, pop_sz = 10):
        self.pop = pop_sz
        self.name = name
        self.content = np.zeros((pop_sz, 1))
        self.stress_matrix = np.zeros((1, pop_sz))
        self.happiness_matrix = np.zeros((1, pop_sz))
        self.no_students = 0
        self.student_arr = np.array([])
        self.highest_stress_contributor = None
        
    def mycontent(self):
        return self.content
    

    
    def my_happiness(self):
        return sum(sum(self.happiness_matrix @ self.content))/2
    
    def my_stress(self):
        return sum(sum(self.stress_matrix @ self.content))/2
    
    def compute_new_stress(self, content):
        return sum(sum(self.stress_matrix @ content))/2

        
    def add_student(self, student):

        self.student_arr = np.append(self.student_arr, student)
        student_row = np.zeros((self.pop,1))
        student.room = self
        student_row[student.name] = 1
       
        self.no_students += 1
        
       
       
        self.stress_matrix =  np.concatenate((self.stress_matrix, 
                                              student.stress_contributed), axis = 0)
        self.happiness_matrix  = np.concatenate((self.happiness_matrix, 
                                                 student.happiness_contributed), axis = 0)

        self.content = np.concatenate((self.content, student_row), axis = 1)
        #to keep track of the student position in the stress and happiness matrix 
        #so that we can easily remove it
        student.pos_in_stress_and_happy_matrix = self.stress_matrix.shape[0] - 1
      
        if (self.no_students > 1):
            self.get_max_stress_contributor()
        
    def get_student(self, student_index):
        return self.student_arr[student_index]
    
    def get_max_stress_contributor(self):
        students_stress = []
        
        for i in range(1, self.no_students):
            mut_content = copy.deepcopy(self.content)
            mut_content[:,i] = 0
            students_stress += [self.compute_new_stress(mut_content)]
            
        std_index = students_stress.index(min(students_stress))
        
        self.highest_stress_contributor = self.student_arr[std_index]
        
    def content_single_mat(self, student):
        std_index = np.where(self.student_arr == student)[0][0] +1
        return self.content[:,std_index]
    
    def get_swap_index(self, room, threshold):
        # if a swap increases hpaiines of both groups make it 
        #add functionality to break down a room , hurestic on stress change
        #
        new_stress_values_self = np.zeros((self.no_students, room.no_students))
        new_happiness_values_self = np.zeros((self.no_students, room.no_students))
        add_counter = 0
        possible_swaps = np.zeros(0)
        old_happiness = self.my_happiness() + room.my_happiness()
        for i in range(self.no_students): # swap each person of self with each person of room 2
            for j in range(room.no_students):
                
                
                # remove person i from self and add person j from self  
                mut_content = copy.deepcopy(self.content)
                mut_content[:, i+1] = room.content_single_mat(room.get_student(j))
                
                room_mut_content = copy.deepcopy(room.content)
                room_mut_content[:, j+1] = self.content_single_mat(self.get_student(i))
                
                
                #compute stress of i given new arrangement
                new_stress_self = self.compute_new_stress(mut_content)
                new_stress_room = self.compute_new_stress(mut_content)

                if(new_stress_self > threshold or new_stress_room> threshold):
                    continue
                else:  #add new stress value to matrix
                    
                    add_counter += 1
                    new_stress_values_self[i, j] = new_stress_self
                    new_happiness_self = self.compute_new_happiness(mut_content) + room.compute_new_happiness(room_mut_content)
                    if new_happiness_self > old_happiness:
                        new_happiness_values_self[i, j] = new_happiness_self
                        
                    #new_happiness_values_self[i, j] = new_happiness_self
                
        max_happiness_index = np.where(new_happiness_values_self == new_happiness_values_self.max())
        post_happiness = np.amax(new_happiness_values_self)
        possible_swaps = []
        #print(max_happiness_index)
        if post_happiness != 0:  
            possible_swaps = [self.get_student(max_happiness_index[0][0]), room.get_student(max_happiness_index[1][0]), post_happiness]
        #ensure that swap only returns when a swap is possible
        return possible_swaps
    def get_swab(self, room, Rooms, total_happiness,threshold):
        roomA_student = self.student_arr
        roomB_student = room.student_arr
        #prior_happiness = self.my_happiness() + room.my_happiness()
        prior_happiness = total_happiness
        max_post_ratio = 0
        max_happiness_group = []
        for student_A in roomA_student:
            for student_B in roomB_student:
                self.remove(student_A)
                room.remove(student_B)
                self.add_student(student_B)
                room.add_student(student_A)
                stress_A = self.my_stress()
                stress_B = room.my_stress()
                post_happiness = overall_happiness(Rooms)
                self.remove(student_B)
                room.remove(student_A)
                self.add_student(student_A)
                room.add_student(student_B)
               
                if stress_A <= threshold and stress_B <= threshold:
                    
                   
                    if post_happiness > prior_happiness and post_happiness > max_post_ratio:
              
                        max_post_ratio = post_happiness
                        max_happiness_group = [student_A, student_B, max_post_ratio]
                
               
        return max_happiness_group
    def modify_content(self, index, new_content):
        self.content[:,index + 1 ] = new_content
    
    def swap(self, my_student, other_student):


        room = other_student.room
        
        self.remove(my_student)
        room.remove(other_student)
        
        self.add_student(other_student)
        room.add_student(my_student)
    
    def compute_new_happiness(self, content):
        return sum(sum(self.happiness_matrix @ content))/2
        
        
    def possible_merge(self, second_room, new_room):

        full_student_array = np.append(self.student_arr , second_room.student_arr)
        for i in full_student_array:
            deep_copied_student = copy.deepcopy(i)
            deep_copied_student.room = new_room
            new_room.add_student(deep_copied_student)
        
        stress = new_room.my_stress()
        happiness = new_room.my_happiness()
        
        return stress, happiness
    
    def merge(self,room2):
        student_arr_to_add = room2.student_arr
        for i in student_arr_to_add:
            self.add_student(i)
    
        
    def remove(self, student):
        student_index = student.name
        student.room = None
        #removing the student from the arr
        self.student_arr = np.array([stu for stu in self.student_arr if stu.name != student.name])
        self.no_students = self.no_students - 1
        #identifying the column index to be removed
        index_column_to_be_removed = list(self.content[student_index]).index(1)
        
        self.content = np.delete(self.content, index_column_to_be_removed, axis = 1)
        stress_and_happ_row_to_remove = student.pos_in_stress_and_happy_matrix 
        
       
        student.pos_in_stress_and_happy_matrix = None
       
        self.stress_matrix = np.delete(self.stress_matrix, stress_and_happ_row_to_remove, axis = 0)
        self.happiness_matrix = np.delete(self.happiness_matrix, stress_and_happ_row_to_remove, axis = 0)
        
        for i in self.student_arr:
            if i.pos_in_stress_and_happy_matrix > stress_and_happ_row_to_remove:
                i.pos_in_stress_and_happy_matrix = i.pos_in_stress_and_happy_matrix-1
                
    def accurate_breaks(self, other_room, threshold, total_happiness, Rooms):
        my_student_arr = self.student_arr
        
        possible_breaks = []
        max_break_group = []
        max_break_happiness = 0
      
        for student in my_student_arr:
            room_of_orgin_happiness = self.my_happiness()
            room_of_orgin_happiness = self.my_stress()
            
            prior_happiness_other_room = other_room.my_happiness()
           
            self.remove(student)
            other_room.add_student(student)
            post_stress = other_room.my_stress()
           
            
          
            
            post_happiness = other_room.my_happiness()
            post_stress = other_room.my_stress()
            post_origin_happiness = self.my_happiness()
            post_count = other_room.no_students
            total_happiness_after_change = overall_happiness(Rooms)
            
            
            
            
            other_room.remove(student)
            self.add_student(student)
           
            if post_stress <= threshold:
              
                if total_happiness_after_change > total_happiness:
                     
                   
                    ratio = total_happiness_after_change
                   
                    if ratio > max_break_happiness:
                        
                        max_break_happiness = ratio
                        print('prior change {%}', total_happiness)
                        print('after change {%}', total_happiness_after_change)
                        
                        
                        max_break_group = [student, other_room,ratio ]
                    
        return max_break_group

    
    
    
    
  


# In[ ]:


#calculating the happiness in rooms
def overall_happiness(Rooms):
    h = 0
    for i in Rooms:
        h+=i.my_happiness()
    
    return h


# In[ ]:


#merging rooms
def room_merge(Rooms, queue):
    top = queue.get().getitem()
    room1 = top[0]
    room2 = top[1]
    room1.merge(room2)
    Rooms = [r for r in Rooms if r.name != room2.name]
    return Rooms

#we will get the swaps between groups
#if lets we have three groups r1, r2, r3 the possible swaps could be be between, (r1,r2), (r1,r3) and (r1,r2)

def add_swaps(q, Rooms, threshold, n):
    h = 0
    for i in Rooms:
        h+=i.my_happiness()
    all_combinations = list(combinations(Rooms, 2))
    for i in all_combinations:
        room1 = i[0]
        room2 = i[1]
       # print('add_swap1')
        best_swab = room1.get_swab(room2,Rooms,h, threshold)
        if len(best_swab) != 0:
           # print('add_swap2')
            student_1 = best_swab[0]
            student_2 = best_swab[1]
            overall_happiness = best_swab[2]
            comb = ComparableCombinations([student_1, student_2], overall_happiness , 'not_application' , 'swap')
            #print('add comb')
            q.put(comb)
    

#adding one from another group to the other that obeys the threshold    
def add_breaks(q, Rooms, threshold, n):
    total_happiness = overall_happiness(Rooms)
    #print(total_happiness)
    for room_i in Rooms:
        for room_j in Rooms:
            if room_i.name != room_j.name:

                
                possible_breaks_forward = room_i.accurate_breaks(room_j, threshold,  total_happiness, Rooms)
                
                if len(possible_breaks_forward) > 0:
                    student = possible_breaks_forward[0]
                    room = possible_breaks_forward[1]
                    post_happiness = possible_breaks_forward[2]
                    print(post_happiness)
                    comb = ComparableCombinations([student, room],post_happiness, 'not_applicable', 'break')
                    q.put(comb)
                
#performing the swaps
#performing the breaks
def perform_break(q, Rooms, broken):
    #print(broken)
    top_comparable_object = q.get()
    top = top_comparable_object.getitem()
    student = top[0]
    room_to_add = top[1]
    orginal_room_of_student = student.room
    forward = (student.name, orginal_room_of_student.name, room_to_add.name)
    backward = (student.name, room_to_add.name,  orginal_room_of_student.name)
    while (forward in broken) or (backward in broken):
        if q.qsize() == 0:
            break
        top_comparable_object = q.get()
        top = top_comparable_object.getitem()
        student = top[0]
        room_to_add = top[1]
        orginal_room_of_student = student.room
    
    if (forward not in broken) and (backward not in broken):
        print('break')
        
 
        orginal_room_of_student.remove(student)
        room_to_add.add_student(student)
        bre = (student.name, orginal_room_of_student.name, room_to_add.name)
        broken.append(bre)
        
      
        

    return Rooms, broken

def perform_break_2(q, Rooms, broken):
    #print(broken)
    print('in break')
    print(overall_happiness(Rooms))
    top_comparable_object = q.get()
    top = top_comparable_object.getitem()
    student = top[0]
    room_to_add = top[1]
    orginal_room_of_student = student.room
    
    while student.name in broken:
        if q.qsize() == 0:
            break
        top_comparable_object = q.get()
        top = top_comparable_object.getitem()
        student = top[0]
        room_to_add = top[1]
        orginal_room_of_student = student.room
    
    if student.name not in broken:
        
        #removing the student from the original room
        orginal_room_of_student.remove(student)
        room_to_add.add_student(student)
       
        broken.append(student.name)
        
      
        

    return Rooms, broken

def perform_the_swaps_2(q, Rooms, swapped):
    
   
    top_comparable_object = q.get()
    top = top_comparable_object.getitem()
   
    student1 = top[0]
    student2 = top[1]
   
    
    candidates_swapped_forward = (student1.name, student2.name)
    candidates_swapped_reversed = (student2.name, student1.name)
    
    while True:
        
        
        forward_yes = False
        reversed_yes = False
        if candidate_swapped_forward in swapped:
            forward_yes = True
        if candidate_swappped_forward in swapped:
            reversed_yes = True
        if not forward_yes and not reversed_yes:
            break
        
        if q.empty():
            break
        top_comparable_object = q.get()
        top = top_comparable_object.getitem()
        student1 = top[0]
        student2 = top[1]
        candidates_swapped_forward = (student1.name, student2.name)
        candidates_swapped_reversed = (student2.name, student1.name)
    
        
    if (candidates_swapped_forward in swapped) or (candidates_swapped_reversed in swapped):
        return Rooms, swapped
        
    
    else:
      
        swapped.append(candidates_swapped_forward)
        swapped.append(candidates_swapped_reversed)
        room1 = student1.room
        room2 = student2.room
        
        room1.remove(student1)
        room2.remove(student2)
        #swapping the students
        room1.add_student(student2)
        room2.add_student(student1)
    
    return Rooms, swapped 
            
    


# In[ ]:


def calculate_stress(g, stress_matrix):
    b = np.sort(g)
    if len(b) == 1:
        return 0
    edges = list(combinations(b,2))
    stress = 0
    for i in edges:
        x = i[0]
        y = i[1]
       
        stress+=stress_matrix[x][y]
    
    return stress



def verify_sol(groups, max_stress, file_name, stress_matrix):
    stress_threshold = max_stress / len(groups.keys())
    for current_room in groups.keys():
        
        group = groups[current_room]
        
        stress = calculate_stress(group, stress_matrix)
        
        if stress > stress_threshold:
            print('error')
            raise Exception(file_name + 'did not obey the thresholds')
        
    


# In[ ]:


def perform_the_swaps(queue, Rooms, swapped):
    
    print(swapped)
    
    new_top = queue.get().getitem()
    
    student1 = new_top[0]
    student2 = new_top[1]
    #getting the rooms
    
    candidates_swapped_forward = (student1.name, student2.name)
    candidates_swapped_reversed = (student2.name, student1.name)
    
    while True:
        print(queue.qsize())
        print('stuck in while loop')
        forward_yes = False
        reversed_yes = False
        if candidates_swapped_forward in swapped:
            forward_yes = True
        if candidates_swapped_reversed in swapped:
            reversed_yes = True
        if not forward_yes and not reversed_yes:
            break
        
        if queue.empty():
            break
        top_comparable_object = queue.get()
        top = top_comparable_object.getitem()
        student1 = top[0]
        student2 = top[1]
        candidates_swapped_forward = (student1.name, student2.name)
        candidates_swapped_reversed = (student2.name, student1.name)
    
        
    if (candidates_swapped_forward in swapped) or (candidates_swapped_reversed in swapped):
        return Rooms, swapped
        
    
    else:
        #print('here at swap')
        swapped.append(candidates_swapped_forward)
        swapped.append(candidates_swapped_reversed)
        room1 = student1.room
        room2 = student2.room
        
        room1.remove(student1)
        room2.remove(student2)
        #swapping the students
        room1.add_student(student2)
        room2.add_student(student1)
    
    return Rooms, swapped
            


# In[ ]:


#to perform solo repeated swaps and breaks

def sole(Rooms, threshold, n):
    copy_of_Rooms =copy.deepcopy(Rooms)
    j = queue.PriorityQueue()
    add_swaps(j, copy_of_Rooms, threshold,n)

    swapped = []
    
    while j.qsize() > 0:
        prev_swapped = copy.deepcopy(swapped)
        copy_of_Rooms, swapped = perform_the_swaps(j,copy_of_Rooms, swapped)
        print(swapped)
        j = queue.PriorityQueue()
        add_swaps(j, Rooms, threshold,n)
        swapped = []
       
    return copy_of_Rooms

def sole_break(Rooms, threshold, n):
    copy_of_Rooms =copy.deepcopy(Rooms)
    j = queue.PriorityQueue()
    add_breaks(j, copy_of_Rooms, threshold,n)
    #print(j.qsize())
    broken = []
    
    while j.qsize() > 0:
        prev_broken = copy.deepcopy(broken)
        copy_of_Rooms, broken = perform_break_2(j,copy_of_Rooms, broken)
        print(broken)
        
        j = queue.PriorityQueue()
        add_breaks(j, Rooms, threshold,n)
        if len(set(prev_broken)) == len(set(broken)):
            break
    return copy_of_Rooms

"""Gets the happiness for people in a room"""
def happinessInGroup(people, happiness_matrix):
    group_happiness = 0
    for i in range(len(people)-1):
        for j in range(i+1, len(people)):
            student1 = people[i]
            student2 = people[j]
            group_happiness += happiness_matrix[student1][student2]
    return group_happiness






def getHappinessTotal(room_arrangement, happiness_matrix):
    
    happiness_across_all_rooms = []
    
    for current_room in room_arrangement.keys():
        people_in_room = room_arrangement[current_room]
        happiness_across_all_rooms.append(happinessInGroup(people_in_room, happiness_matrix))
            
    
    return sum(happiness_across_all_rooms)

def read_files(file_name, n):
    number_of_students = 0
    max_stress = 0
    student_input = n
    happiness_arr = [[0 for x in range(student_input)] for y in range(student_input)]
    stress_arr = [[0 for x in range(student_input)] for y in range(student_input)]
    with open( file_name,"r") as fp:
        current_line = fp.readline()
        number_of_students = int(current_line.strip())
        current_line = fp.readline()
        max_stress = float(current_line.strip())
        current_line = fp.readline()
        while current_line:
            i,j,hapiness,stress = current_line.split(" ")
            i = int(i)
            j = int(j)
            hapiness = float(hapiness)
            stress = float(stress)
        
        
            happiness_arr[i][j] = hapiness
            happiness_arr[j][i] = hapiness
        
            stress_arr[i][j] = stress
            stress_arr[j][i] = stress
        
            current_line = fp.readline()
    
    return stress_arr, happiness_arr, max_stress
    

def safe_output(file_location, file_name, groups):
    direct = file_location + '/' + file_name
    order = {}
    
    for i in groups.keys():
        nodes = groups[i]
        for j in nodes:
            order[j] = i
    print(order)
    nodes_Sorted = np.sort(list(order.keys()))
    f2 = open(direct +'.out', "w")

    for node in nodes_Sorted:
        room = order[node]
        string = str(node) + "  " + str(room)
        f2.write(string)
        f2.write('\n')
    f2.close()


# In[ ]:


def repeat_merge(Rooms, threshold, n):
    has_groups_to_be_formed_under_threshold = True
    q = queue.PriorityQueue() 
    get_merges(Rooms, q, threshold, n)
    print("repeat")
    if q.qsize() == 0:
        has_groups_to_be_formed_under_threshold = False
        while has_groups_to_be_formed_under_threshold:
            if q.qsize() > 0:
                Rooms = room_merge(Rooms, q)
            q = queue.PriorityQueue() 
            get_merges(Rooms, q, threshold, n)
            if q.qsize() == 0:
                has_groups_to_be_formed_under_threshold = False
    return Rooms
        


# In[ ]:


def greedy_solver_algorithm_three(stress_matrix, happiness_matrix, n, Smax,iteration, starting_k = 35, repetitions = 20):
    
    #creating students, creating rooms for each students, and adding them appropriately
    Rooms = []
    valid_solutions = {}
    for i in np.arange(n):
        room = Room('Room' + str(i+1), n)
        student = Student(i, stress_matrix, happiness_matrix)
        room.add_student(student)
        Rooms.append(room)
    
    #performing merge from k = n to k = ? till possible
    
    valid_solutions[n] = Rooms
    for k in range(1, starting_k)[::-1]:
        print(k)
        print('room_len {%}',len(Rooms))
        threshold = Smax/k
      
        
        has_groups_to_be_formed_under_threshold = True
        q = queue.PriorityQueue() 
        get_merges(Rooms, q, threshold, n)
        if q.qsize() == 0:
            has_groups_to_be_formed_under_threshold = False
        while has_groups_to_be_formed_under_threshold:
            if q.qsize() > 0:
                Rooms = room_merge(Rooms, q)
                
            q = queue.PriorityQueue() 
            get_merges(Rooms, q, threshold, n)
            for i in Rooms:
                #print(i.content)
                print(i.my_stress())
                print([j.name for j in i.student_arr])
                print([j.room.name for j in i.student_arr])
            if q.qsize() == 0:
                has_groups_to_be_formed_under_threshold = False
        
        if len(Rooms) == k:
            valid_solutions[k] = copy.deepcopy(Rooms)
        
#         conducting the swaps and breaks given the threshold
       
        post_iterate = False
        if True:
       
            swapped = []
            broken = []
           
            prev_swap  = swapped
            new_broken = []
            for i in np.arange(iteration):
                
                prev = copy.copy(broken)
                choice = np.random.choice(['T', 'H'], p =[0,1])
                
                if choice =='H':
                  
                    m = queue.PriorityQueue()
                    add_breaks(m, Rooms, threshold, n)
                    if m.qsize() == 0:
                        break
                    if m.qsize() > 0:
                        post_iterate = True
                       
                        Rooms, broken = perform_break_2(m, Rooms, broken)
                        
                        broken = []
                        Rooms = [r for r in Rooms if r.no_students != 0]
                        
                  
                if i%5 == 0:
                    print('here')
                    Rooms = repeat_merge(Rooms, threshold, n)
                if len(Rooms) == k:
                   
                    Rooms = [r for r in Rooms if r.no_students != 0]
                    valid_solutions[k] = copy.deepcopy(Rooms)
          
                     
            
        if post_iterate:
            Rooms = repeat_merge(Rooms, threshold, n)
        if len(Rooms) == k:
            Rooms = [r for r in Rooms if r.no_students != 0]
            valid_solutions[k] = copy.deepcopy(Rooms)
                
                
                
#                
        
        print("post_iter")
        print('boundary {%}', k)
        print('Rlen{%}', len(Rooms))
        if len(Rooms) == k:
            for i in np.arange(repetitions):
                print('repetition {%}',i)
                Rooms = sole(Rooms, threshold , n)
            
            for i in np.arange(repetitions):
                print('breaking when equal')
                Rooms = sole_break(Rooms, threshold, n)
                Rooms = [r for r in Rooms if r.no_students != 0]
            
            valid_solutions[k] = copy.deepcopy(Rooms)
            
           
        
        for i in Rooms:
                #print(i.content)
                
                print(i.my_stress())
                print([j.name for j in i.student_arr])
                print([j.room.name for j in i.student_arr])
        print("post_iter_done")
       
    
    #once all the iterations are done, we can then return the best solution
    print(valid_solutions.keys())
    print(min(valid_solutions.keys()))
    key = min(valid_solutions.keys())
    best_sol = valid_solutions[key]
  
    return best_sol


# In[ ]:


""" Parses the input in the form of aan adjency list"""
""" Taken from our SolutionVerifier """
number_of_students = 0
max_stress = 0
student_input = 50
happiness_arr = [[0 for x in range(student_input)] for y in range(student_input)]
stress_arr = [[0 for x in range(student_input)] for y in range(student_input)]

with open('inputs/large-95' + ".in","r") as fp:
    
    current_line = fp.readline()

    number_of_students = int(current_line.strip())
    
    current_line = fp.readline()
    max_stress = float(current_line.strip()) 
    current_line = fp.readline()

    while current_line:
        i,j,hapiness,stress = current_line.split(" ")
        
        i = int(i)
        j = int(j)
        hapiness = float(hapiness)
        
        
        stress = float(stress)
        #if stress > max_stress/12:
            #stress = 100
        
        
        happiness_arr[i][j] = hapiness 
        happiness_arr[j][i] = hapiness 
        
        stress_arr[i][j] = stress 
        stress_arr[j][i] = stress 
        
        current_line = fp.readline()
    


# In[ ]:


#you can change the starting k parameter, set something close to the current groups probably\
#you can tweak merge - when you look at the get merge function, we are putting in the combinations into the queue 
#with a value -stress,you can change as you wish like maybe happiness, -happiness, our queue outputs a max by nature, 
#if you want it output the min, then you can just negate any value like how i negated stress
sol3 = greedy_solver_algorithm_three(stress_arr, happiness_arr, 50, max_stress,10000000,5, 20)


# In[ ]:




