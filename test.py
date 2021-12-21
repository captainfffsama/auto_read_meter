import core.utils as u

pt1=(0,0)
pt2=(1,1)
pt3=(0,1)
pt4=(0,2)
print(u.get_angle(*pt1,*pt2))
print(u.get_angle(*pt1,*pt2,True))
print(u.get_angle(*pt3,*pt2))
print(u.get_angle(*pt3,*pt2,True))
print(u.get_angle(*pt4,*pt2,True))