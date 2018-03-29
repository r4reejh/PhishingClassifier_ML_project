l = [85,91,-1,23,0]
count = 0


def swap(index1,index2):
	temp = l[index1]
	l[index1]=l[index2]
	l[index2]=temp
	
for i in range(0,len(l)):
	for j in range(i+1,len(l)):
		if l[i]>=l[j]:
			swap(i,j)

print l	
