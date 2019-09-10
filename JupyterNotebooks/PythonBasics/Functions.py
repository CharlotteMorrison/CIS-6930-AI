#!/usr/bin/env python
# coding: utf-8

# # Functions
# Syntax:
# ```python
# def func(parameter1, parameter2=default_value):
#     """
#     Doc String
#     """
#     <code block>
#     return value
# ```

# In[1]:


def Fib (n=1000):
    """ Compute the sequence of Fibonacci numbers up to n """
    a, b = 1, 1
    while (a < n):
        print (a, end=" ")
        a, b = b, a+b

if __name__ == "__main__":
    Fib()
    print ()
    Fib(10000)
    print ()
    Fib(n=10)


# # Default Argument Values
# The most useful form is to specify a default value for one or more arguments. This creates a function that can be called with fewer arguments than it is defined to allow.

# In[3]:


def ask_ok(prompt, retries=4, complaint='Yes or no, please!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise IOError('refusenik user')
        print (complaint)
        
if __name__ == "__main__":
    print (ask_ok("Do you want to go?"))


# # Keyword Arguments
# Functions can also be called using keyword arguments of the form *kwarg=value*.
# In a function definition and a function call, keyword arguments **must** follow positional arguments.

# In[4]:


def myPrint(b, a='Hello', c=100):
    print (a, end=" ")
    print (b, end=" ")
    print (c)
    
if __name__ == "__main__":
    myPrint(b=300)
    #myPrint(9, b='World')
    #myPrint(88, c='world', b='hello')
    #myPrint(b='What', 10000)


# # Pass Function Names to Other Functions

# In[5]:


def f1(a=100):
    print ("I saved %d dollars today." % a)
    
def f2(a=100):
    print ("I spent %d dollars today." % a)
    
def F(greetings='Hi everyone!', f=f1):
    print (greetings)
    f(8)
    
if __name__ == "__main__":
    F()
    F(f=f2)


# # Built in Functions

# In[6]:


if __name__ == "__main__":
    ratings = {'chocolate':2.5, 'strawberry':4, 'vanilla':4.5}
    
    print (len(ratings))
    print (sum([pow(v,2) for (k,v) in ratings.items()]))

