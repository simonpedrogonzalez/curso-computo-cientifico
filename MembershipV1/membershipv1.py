import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import numpy.linalg as lin
import scipy.special as spc
import scipy.optimize as opt
import scipy.cluster.hierarchy as shc 
import sklearn.cluster as clu
plt.interactive(True)

#
#cargo la tabla con: Ra,Dec,Plx,ePlx,pmRa,epmRa,pmDec,epmDec,epsi,G,BP-RP,RV,eRv
#

tabla= np.genfromtxt('ic2395.csv',delimiter=',',skip_header=1,names=None, dtype="f8", usecols=(1,2,6,7,8,9,10,11,12,16,19,20,21))


####################################################################################
#
#Paso1
#


#graficamos toda la muestra en el espacio de movimiento propio y paralaje

tabla2=np.where(tabla[:,2]>0)
tabla2=tabla[tabla2]
bines=np.int64(round(np.sqrt(len(tabla2))+0.5))
his=np.histogram(tabla2[:,2],bins=bines,density=True)
rango=np.max(tabla2[:,2])-np.min(tabla2[:,2])
fac=rango/bines
frec=his[0]*fac
plt.bar(his[1][1:]-fac/2,frec,0.25) 
plt.title('Distribución de la paralaje de la población')
plt.xlabel('Parallax (mas)')
plt.ylabel('n')


plt.figure()
plt.plot(tabla[:,4],tabla[:,6],'b.',label='Toda la muestra')
plt.xlabel('Proper motion Ra')
plt.ylabel('Proper motion Dec')
plt.title('Espacio de movimiento propio de toda la población y de la muestra (G < 14mag)')



#
# Seleccionamos objetos con G<14 y Plx positivo para poder ver al cúmulo en los espacios
#

tablaG=np.where((tabla[:,9]<=14) & (tabla[:,2]>=0))

tablaG=tabla[tablaG]


plt.plot(tablaG[:,4],tablaG[:,6],'r.',label='G < 14mag')
plt.legend()



#Grafico la distribución de la paralaje

bines=np.int64(round(np.sqrt(len(tablaG))+0.5)) 
his=np.histogram(tablaG[:,2],range=(0,2),bins=30,density=True)
rango=np.max(tablaG[:,2])-np.min(tablaG[:,2])
fac=2/30 
frec=his[0]*fac

plt.figure()
plt.bar(his[1][1:]-fac/2,frec,0.06) 
plt.title('Distribución de la paralaje de la muestra')
plt.xlabel('Parallax (mas)')
plt.ylabel('n')


###########################################################################################
#
#Paso 2
#

#Ajuste de la distribución de paralaje utilizando el método de Levenberg-Marquardt

his2,xx2=np.histogram(tablaG[:,2],range=(0,2),bins=30,density=True)
his2=his2/np.sum(his2) 
xx2=xx2-fac/2

def funcajus(par,xx,yy): 
             zz=(  (  (par[4]/(np.sqrt(2.*np.pi*par[1])))*(np.exp(-((xx-par[0])**2/(2*(par[1])**2)))  ))  +  (  (par[5]/(np.sqrt(2.*np.pi*par[3])))*(np.exp(-((xx-par[2])**2/(2*(par[3])**2))))  )  )-yy 
             return zz 

solu=opt.least_squares(funcajus,np.array([0.2,0.25,1.36,0.02,0.09,0.06]),method='lm',ftol=1.e-12,gtol=1.e-12,xtol=1.e-12,args=(xx2[1:],his2)) 
solu=solu['x'] 

px= ((solu[4]/(np.sqrt(2.*np.pi*solu[1])))*(np.exp(-((xx2-solu[0])**2/(2*(solu[1])**2)))))+( (solu[5]/(np.sqrt(2.*np.pi*solu[3])))*(np.exp(-((xx2-solu[2])**2/(2*(solu[3])**2))))  )


plt.figure()
plt.bar(his[1][1:]-fac/2,frec,0.06,label='Distribución de la paralaje de la muestra') 
plt.title('Ajuste de la distribución de la paralaje de la muestra')
plt.xlabel('Parallax (mas)')
plt.ylabel('n')
plt.plot(xx2,px,'r-',label='Ajuste Levenberg-Marquardt')
plt.legend()

print('Parámetros de salida del ajuste de Levenberg-Marquardt',solu)


#Medición del movimiento propio del cúmulo 


#solamente tenemos una tabla con los movimientos propios
pm=np.vstack((tablaG[:,4],tablaG[:,6])) 
#Necesitamos que sean dos filas 
pmT=pm.T
plt.figure()
dend=shc.dendrogram(shc.linkage(pmT,method='ward'),truncate_mode='lastp',p=100,no_labels=True) 
plt.axhline(y=8,color='b',linestyle='--')
plt.title('Dendrograma de movimiento propio con 100 nodos')

#Ahora vemos qué objeto pertenece a cuál grupo

grupos=clu.AgglomerativeClustering(n_clusters=75,affinity='euclidean',linkage='ward')
cl=grupos.fit_predict(pmT)
print('Determinación de la pertenencia de cada objeto',cl)
plt.figure()
plt.scatter(pmT[:,0],pmT[:,1],c=grupos.labels_,alpha=0.5,marker='.')
plt.xlabel('Proper motion Ra')
plt.ylabel('Proper motion Dec')
plt.title('Agrupamiento jerárquico en movimiento propio')
cumulos=tabla[cl]
grupos=np.column_stack((tablaG,cl))
#Busco cuál es el número del cúmulo teniendo en cuenta el centro del cúmulo hallado gráficamente

np.where((grupos[:,4]<-4.28) & (grupos[:,4]>-4.48)&(grupos[:,6]>3.16)&(grupos[:,6]<3.36)) 
np.where((grupos[:,4]<-4.28) & (grupos[:,4]>-4.48)&(grupos[:,6]>3.16)&(grupos[:,6]<3.36))
#Con ese comando puedo verificar si todos los objetos poseen el mismo rótulo: grupos[459,:]
print('IC2395 es el grupo 26 ')
#Una vez identificado el rótulo del grupo selecciono todos aquellos objetos que coinciden

mmp=np.where((grupos[:,13])==26)
mmp=tablaG[mmp]

#Valores medios y desviaciones estándar de movimiento propio

meanmura=np.mean(mmp[:,4],axis=0)
stdmura=np.std(mmp[:,4],axis=0,ddof=0)
meanmudec=np.mean(mmp[:,6],axis=0)
stdmudec=np.std(mmp[:,6],axis=0,ddof=0)
meanplx=solu[2]
stdplx=solu[3]
print('Movimiento propio medio en RA',meanmura)
print('Desviación estándar del movimiento propio en RA',stdmura)
print('Movimiento propio medio en DEC',meanmudec)
print('Desviación estándar del movimiento propio en DEC',stdmudec)
print('Paralaje media en RA',solu[2])
print('Desviación estándar de la paralaje',solu[3])





###########################################################################################
#
#Paso 3
#

#Calculamos la distancia de los objetos al centro del cúmulo

distmu=np.sqrt(((tabla[:,4]+4.389)**2)+((tabla[:,6]-3.268)**2))
tabla=np.column_stack((tabla,distmu)) 

#Luego seleccionamos a aquellos que se encuentran a 2sigmas de la paralaje media y cuya distancia sea inferior a 1.1


plxmu=np.where(    (tabla[:,2]>(meanplx-2*stdplx)) & (tabla[:,2]<(meanplx+2*stdplx)) & (tabla[:,13]<1.11)     )
plxmu=tabla[plxmu]

plxmu2=np.where(   (tabla[:,2]>(meanplx-1*stdplx)) & (tabla[:,2]<(meanplx+1*stdplx)) & (tabla[:,13]<0.56  )       )
plxmu2=tabla[plxmu2]

#Hacer el filtro en epsi y en error de paralaje

plxerror=plxmu[:,3]/plxmu[:,2]
plxmuu=np.column_stack((plxmu,plxerror))
plxmuuu=np.where((plxmuu[:,14]<0.1))
plxmuu=plxmu[plxmuuu]

plxerror2=plxmu2[:,3]/plxmu2[:,2]
plxmuu2=np.column_stack((plxmu2,plxerror2))
plxmuuu2=np.where((plxmuu2[:,14]<0.1))
plxmuu2=plxmu2[plxmuuu2]



print('Cantidad de probables miembros del cúmulo es: 2sigma=',len(plxmu),', 1sigma=',len(plxmu2))


#Finalmente graficamos los resultados

plt.figure()
plt.plot(tabla[:,4],tabla[:,6],'r.',label='Toda la población')
plt.plot(tablaG[:,4],tablaG[:,6],'g.',label='Objetos con G<14mag')
plt.plot(plxmu[:,4],plxmu[:,6],'b.',label='Probables miembros 2sigma')
plt.plot(plxmu2[:,4],plxmu2[:,6],'k.',label='Probables miembros 1sigma')
plt.plot(plxmuu[:,4],plxmuu[:,6],'y.',label='Probables miembros 2sigma(error)')
plt.plot(plxmuu2[:,4],plxmuu2[:,6],'c.',label='Probables miembros 1sigma(error)')
plt.xlabel('Proper motion RA (mas/yr)')
plt.ylabel('Proper motion DEC (mas/yr)')
plt.title('Espacio de movimiento propio')
plt.legend()


plt.figure()
plt.plot(tabla[:,10],tabla[:,9],'r.',label='Toda la población')
plt.plot(tablaG[:,10],tablaG[:,9],'g.',label='Objetos con G<14mag')
plt.plot(plxmu[:,10],plxmu[:,9],'bo',label='Probables miembros 2sigma')
plt.plot(plxmu2[:,10],plxmu2[:,9],'ko',label='Probables miembros 1sigma')
plt.plot(plxmuu[:,10],plxmuu[:,9],'y.',label='Probables miembros 2sigma(error)')
plt.plot(plxmuu2[:,10],plxmuu2[:,9],'c.',label='Probables miembros 1sigma(error)')
#Para invertir el eje y se utiliza el siguiente comando plt.axis([x1,x2,y1,y2])
plt.axis([-1.5,6,22,2]) 
plt.xlabel('BP-RP (mag)')
plt.ylabel('G(mag)')
plt.title('CMD')
plt.legend()



plt.figure()
plt.plot(tabla[:,3],tabla[:,2],'r.',label='Toda la población')
plt.plot(tablaG[:,3],tablaG[:,2],'g.',label='Objetos con G<14mag')
plt.plot(plxmu[:,3],plxmu[:,2],'bo',label='Probables miembros 2sigma')
plt.plot(plxmu2[:,3],plxmu2[:,2],'ko',label='Probables miembros 1sigma')
plt.plot(plxmuu[:,3],plxmuu[:,2],'y.',label='Probables miembros 2sigma(error)')
plt.plot(plxmuu2[:,3],plxmuu2[:,2],'c.',label='Probables miembros 1sigma(error)')
plt.xlabel('Error paralaje (mas)')
plt.ylabel('Paralaje(mas)')
plt.title('Espacio de paralaje')
plt.legend()



######Falta agregar los parámetros finales y que los muestre con un comando print


meanmurac=np.mean(plxmu[:,4],axis=0)
stdmurac=np.std(plxmu[:,4],axis=0,ddof=0)
meanmudecc=np.mean(plxmu[:,6],axis=0)
stdmudecc=np.std(plxmu[:,6],axis=0,ddof=0)
meanplxc=np.mean(plxmu[:,2],axis=0)
stdplxc=np.std(plxmu[:,2],axis=0)



meanmurac2=np.mean(plxmu2[:,4],axis=0)
stdmurac2=np.std(plxmu2[:,4],axis=0,ddof=0)
meanmudecc2=np.mean(plxmu2[:,6],axis=0)
stdmudecc2=np.std(plxmu2[:,6],axis=0,ddof=0)
meanplxc2=np.mean(plxmu2[:,2],axis=0)
stdplxc2=np.std(plxmu2[:,2],axis=0)


meanmuracu=np.mean(plxmuu[:,4],axis=0)
stdmuracu=np.std(plxmuu[:,4],axis=0,ddof=0)
meanmudeccu=np.mean(plxmuu[:,6],axis=0)
stdmudeccu=np.std(plxmuu[:,6],axis=0,ddof=0)
meanplxcu=np.mean(plxmuu[:,2],axis=0)
stdplxcu=np.std(plxmuu[:,2],axis=0)


meanmuracuu=np.mean(plxmuu2[:,4],axis=0)
stdmuracuu=np.std(plxmuu2[:,4],axis=0,ddof=0)
meanmudeccuu=np.mean(plxmuu2[:,6],axis=0)
stdmudeccuu=np.std(plxmuu2[:,6],axis=0,ddof=0)
meanplxcuu=np.mean(plxmuu2[:,2],axis=0)
stdplxcuu=np.std(plxmuu2[:,2],axis=0)



print('Movimiento propio medio en RA del cúmulo: 1sigma',meanmurac2,', 2sigma',meanmurac,', 1simga (corr)',meanmuracuu,', 2sigma(corr)',meanmuracu)
print('Desviación estándar del movimiento propio en RA del cúmulo: 1sigma',stdmurac2,', 2sigma',stdmurac,', 1simga (corr)',stdmuracuu,', 2sigma(corr)',stdmuracu)
print('Movimiento propio medio en DEC del cúmulo: 1sigma',meanmudecc2,', 2sigma',meanmudecc,', 1simga (corr)',meanmudeccuu,', 2sigma(corr)',meanmudeccu)
print('Desviación estándar del movimiento propio en DEC del cúmulo: 1sigma',stdmudecc2,', 2sigma',stdmudecc,', 1simga (corr)',stdmudeccuu,', 2sigma(corr)',stdmudeccu)
print('Paralaje media (sin corrección 0.03mas): 1sigma',meanplxc2,', 2sigma',meanplxc,', 1simga (corr)',meanplxcuu,', 2sigma(corr)',meanplxcu)
print('Desviación estándar de la paralaje (sin corrección): 1sigma',stdplxc2,', 2sigma',stdplxc,', 1simga (corr)',stdplxcuu,', 2sigma(corr)',stdplxcu)
print('Cantidad de probables miembros del cúmulo es: 1sigma=',len(plxmu2),', 2sigma=',len(plxmu),',1 sigma (error)',len(plxmuu2),',2sigma(error)',len(plxmuu))


np.savetxt('2sigma.txt',plxmu,delimiter=' ')
np.savetxt('1sigma.txt',plxmu2,delimiter=' ')
np.savetxt('2sigmae.txt',plxmuu,delimiter=' ')
np.savetxt('1sigmae.txt',plxmuu2,delimiter=' ')








