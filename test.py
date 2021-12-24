from typing import overload


class Q:
    def __init__(self):
        self.s_n=1

    def p(self):
        print(self.s_n)
    

class Test(Q):
    def __init__(self):
        self.s_n=2


def main():
    test=Test()
    test.p()
if __name__=="__main__":
    main()

