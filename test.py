from typing import overload


class Q:
    def add(self,a,b):
        return a+b

    def test(self):
        func=getattr(self,"add")
        return func(1,2)



def main():
    test=Q()
    print(test.test())
    breakpoint()
if __name__=="__main__":
    main()

