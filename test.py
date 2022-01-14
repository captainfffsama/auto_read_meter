from typing import overload
from test_param import get_args


class Q:
    def add(self,a,b):
        return a+b

    def test(self):
        func=getattr(self,"add")
        return func(1,2)



def main():
    test=Q()
    print(test.test())
    args=get_args()
    print(args["a"])
    breakpoint()
if __name__=="__main__":
    main()

