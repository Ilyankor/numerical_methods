def taylor2(g,gt,gy,h,n,t0,y0):
    y=y0
    t=t0
    for i in range(n):
        y=y+g(t,y)*h+0.5*(gt(t,y)+gy(t,y)*g(t,y))*h**2
        t=t+h
    return y

def taylor3(g,gt,gy,gty,gyy,gtt,h,n,t0,y0):
    y=y0
    t=t0
    for i in range(n):
        z1=gtt(t,y)+2.0*gty(t,y)*g(t,y)+gyy(t,y)*g(t,y)**2
        z2=gy(t,y)*(gt(t,y)+gy(t,y)*g(t,y))
        y=y+g(t,y)*h+0.5*(gt(t,y)+gy(t,y)*g(t,y))*h**2+(z1+z2)*h**3/6.0
        t=t+h
    return y