! The wrapper f2py compiles FORTRAN code in a method
! Python understands. The f2py... "comments" are
! indicators for this, telling f2py what is an input
! and what is an output. In this case, call the
! function from Python (after import epv) with
! q = epv.epv(u, v, t, Zeta, p, lat, lon)
      subroutine epv(q,u,v,t,Z,p,lat,lon,nt,nz,ny,nx)
      implicit none
!f2py intent(out) q
!f2py intent(in) u,v,t,Z,p,lat,lon,nt,nz,ny,nx
      integer nt,nz,ny,nx
      real q(nt,nz,ny,nx)   ! Potential vorticity, output
      real u(nt,nz,ny,nx)   ! zonal wind component
      real v(nt,nz,ny,nx)   ! meridional wind component
      real t(nt,nz,ny,nx)   ! Potential temperature
      real Z(nt,nz,ny,nx)   ! Relative vorticity
      real p(nz)            ! pressure [Pa]
      real lat(ny)
      real lon(nx)
      real pi,a,Om,g,dTdp,dTdx,dTdy,dudp,dvdp,dxi,f
      real dx,dy,dp1,dp2
      integer n,k,j,i
      pi=2.0*asin(1.0)
      a=6.37e6
      Om=7.292e-5
      g=9.81
      dxi=lon(2)-lon(1)
      do n=1,nt
        do k=1,nz
         if( k .eq. 1 ) then
           dp2 = p(k)-p(k+1)
         else if( k .eq. nz ) then
           dp1 = p(k-1)-p(k)
         else
           dp1 = p(k-1)-p(k)
           dp2 = p(k)-p(k+1)
         end if
         do j=1,ny
            f=2.0*Om*sin(lat(j))
            do i=1,nx
             dx = dxi*a*cos(lat(j))

             ! Pressure differential
             if( k .eq. 1 ) then !dp1==0
               dTdp = (T(n,k,j,i)-T(n,k+1,j,i))/(dp2)
               dudp = (u(n,k,j,i)-u(n,k+1,j,i))/(dp2)
               dvdp = (v(n,k,j,i)-v(n,k+1,j,i))/(dp2)
             else if( k .eq. nz ) then !dp2==0
               dTdp = (T(n,k-1,j,i)-T(n,k,j,i))/(dp1)
               dudp = (u(n,k-1,j,i)-u(n,k,j,i))/(dp1)
               dvdp = (v(n,k-1,j,i)-v(n,k,j,i))/(dp1)
             else
               dTdp = (dp2*T(n,k-1,j,i)+(dp1-dp2)*T(n,k,j,i)
     $              -dp1*T(n,k+1,j,i))/(2.0*dp1*dp2)
               dudp = (dp2*u(n,k-1,j,i)+(dp1-dp2)*u(n,k,j,i)
     $              -dp1*u(n,k+1,j,i))/(2.0*dp1*dp2)
               dvdp = (dp2*v(n,k-1,j,i)+(dp1-dp2)*v(n,k,j,i)
     $              -dp1*v(n,k+1,j,i))/(2.0*dp1*dp2)
             end if

             ! Derivative in X-direction
             if( i.eq.1 )then
               dTdx = (T(n,k,j,2)-T(n,k,j,nx))/(dx)
             else if( i.eq.nx ) then
               dTdx = (T(n,k,j,1)-T(n,k,j,nx-1))/(dx)
             else
               dTdx = (T(n,k,j,i+1)-T(n,k,j,i-1))/(dx)
             end if

             ! Derivative in Y-direction
             if( j .eq. 1 ) then
               dy   = (lat(j+1)-lat(1))*a
               dTdy = (T(n,k,j+2,i)-T(n,k,j,i))/(2.0*dy)
             else if( j .eq. ny ) then
               dy =(lat(j)-lat(j-1))*a
               dTdy = (T(n,k,j,i)-T(n,k,j-2,i))/(2.0*dy)
             else
               dy =(lat(j+1)-lat(j-1))*a
               dTdy = (T(n,k,j+1,i)-T(n,k,j-1,i))/(dy)
             end if

             q(n,k,j,i) = -g*((Z(n,k,j,i)+f)*dTdp+dudp*dTdy-dvdp*dTdx)
            end do
          end do
        end do
      end do
      return
      end

      subroutine epv3d(q,u,v,t,Z,p,lat,lon,nz,ny,nx)
      implicit none
!f2py intent(out) q
!f2py intent(in) u,v,t,Z,p,lat,lon,nz,ny,nx
      integer nz,ny,nx
      real q(nz,ny,nx)
      real u(nz,ny,nx)
      real v(nz,ny,nx)
      real t(nz,ny,nx)
      real Z(nz,ny,nx)
      real p(nz)
      real lat(ny)
      real lon(nx)
      real pi,a,Om,g,dTdp,dTdx,dTdy,dudp,dvdp,dxi,f
      real dx,dy,dp1,dp2
      integer k,j,i
      pi=2.0*asin(1.0)
      a=6.37e6
      Om=7.292e-5
      g=9.81
      dxi=lon(2)-lon(1)
      do k=1,nz
       if( k .eq. 1 ) then
         dp2 = p(k)-p(k+1)
       else if( k .eq. nz ) then
         dp1 = p(k-1)-p(k)
       else
         dp1 = p(k-1)-p(k)
         dp2 = p(k)-p(k+1)
       end if
       do j=1,ny
          f=2.0*Om*sin(lat(j))
          do i=1,nx
           dx = dxi*a*cos(lat(j))

           ! Pressure differential
           if( k .eq. 1 ) then !dp1==0
             dTdp = (T(k,j,i)-T(k+1,j,i))/(dp2)
             dudp = (u(k,j,i)-u(k+1,j,i))/(dp2)
             dvdp = (v(k,j,i)-v(k+1,j,i))/(dp2)
           else if( k .eq. nz ) then !dp2==0
             dTdp = (T(k-1,j,i)-T(k,j,i))/(dp1)
             dudp = (u(k-1,j,i)-u(k,j,i))/(dp1)
             dvdp = (v(k-1,j,i)-v(k,j,i))/(dp1)
           else
             dTdp = (dp2*T(k-1,j,i)+(dp1-dp2)*T(k,j,i)
     $            -dp1*T(k+1,j,i))/(2.0*dp1*dp2)
             dudp = (dp2*u(k-1,j,i)+(dp1-dp2)*u(k,j,i)
     $            -dp1*u(k+1,j,i))/(2.0*dp1*dp2)
             dvdp = (dp2*v(k-1,j,i)+(dp1-dp2)*v(k,j,i)
     $            -dp1*v(k+1,j,i))/(2.0*dp1*dp2)
           end if

           ! Derivative in X-direction
           if( i.eq.1 )then
             dTdx = (T(k,j,2)-T(k,j,nx))/(dx)
           else if( i.eq.nx ) then
             dTdx = (T(k,j,1)-T(k,j,nx-1))/(dx)
           else
             dTdx = (T(k,j,i+1)-T(k,j,i-1))/(dx)
           end if

           ! Derivative in Y-direction
           if( j .eq. 1 ) then
             dy   = (lat(j+1)-lat(1))*a
             dTdy = (T(k,j+2,i)-T(k,j,i))/(2.0*dy)
           else if( j .eq. ny ) then
             dy =(lat(j)-lat(j-1))*a
             dTdy = (T(k,j,i)-T(k,j-2,i))/(2.0*dy)
           else
             dy =(lat(j+1)-lat(j-1))*a
             dTdy = (T(k,j+1,i)-T(k,j-1,i))/(dy)
           end if

           q(k,j,i) = -g*((Z(k,j,i)+f)*dTdp+dudp*dTdy-dvdp*dTdx)
          end do
        end do
      end do
      return
      end

      subroutine rel_vort(Z,u,v,lat,lon,nt,nz,ny,nx)
! Calculates relative vorticity
! Z = dU/dx - dV/dy
      implicit none
!f2py intent(out) Z
!f2py intent(in) u,v,lat,lon,nt,nz,ny,nx
      integer nt,nz,ny,nx
      real Z(nt,nz,ny,nx)
      real u(nt,nz,ny,nx)
      real v(nt,nz,ny,nx)
      real lat(ny)
      real lon(nx)
      real a,dVdx,dUdy,dxi
      real dx,dy
      integer n,k,j,i

      dxi=lon(2)-lon(1)
      a=6.37e6
      do n=1,nt
        do k=1,nz
         do j=1,ny
            do i=1,nx
             dx = dxi*a*cos(lat(j))
             ! Derivative in X-direction
             if( i.eq.1 )then
               dVdx = (v(n,k,j,2)-v(n,k,j,nx))/(dx)
             else if( i.eq.nx ) then
               dVdx = (v(n,k,j,1)-v(n,k,j,nx-1))/(dx)
             else
               dVdx = (v(n,k,j,i+1)-v(n,k,j,i-1))/(dx)
             end if

             ! Derivative in Y-direction
             if( j .eq. 1 ) then
               dy   = (lat(j+1)-lat(1))*a
               dUdy = (u(n,k,j+1,i)-u(n,k,j,i))/(dy)
             else if( j .eq. ny ) then
               dy =(lat(j)-lat(j-1))*a
               dUdy = (u(n,k,j,i)-u(n,k,j-1,i))/(dy)
             else
               dy =(lat(j+1)-lat(j-1))*a
               dUdy = (u(n,k,j+1,i)-u(n,k,j-1,i))/(dy)
             end if
             Z(n,k,j,i)=dVdx-dUdy
            end do
          end do
        end do
      end do
      return
      end

      subroutine rel_vort3d(Z,u,v,lat,lon,nz,ny,nx)
! Calculates relative vorticity
! Z = dU/dx - dV/dy
      implicit none
!f2py intent(out) Z
!f2py intent(in) u,v,lat,lon,nt,nz,ny,nx
      integer nz,ny,nx
      real Z(nz,ny,nx)
      real u(nz,ny,nx)
      real v(nz,ny,nx)
      real lat(ny)
      real lon(nx)
      real a,dVdx,dUdy,dxi
      real dx,dy
      integer k,j,i

      dxi=lon(2)-lon(1)
      a=6.37e6
      do k=1,nz
        do j=1,ny
          do i=1,nx
           dx = dxi*a*cos(lat(j))
           ! Derivative in X-direction
           if( i.eq.1 )then
             dVdx = (v(k,j,2)-v(k,j,nx))/(dx)
           else if( i.eq.nx ) then
             dVdx = (v(k,j,1)-v(k,j,nx-1))/(dx)
           else
             dVdx = (v(k,j,i+1)-v(k,j,i-1))/(dx)
           end if

           ! Derivative in Y-direction
           if( j .eq. 1 ) then
             dy   = (lat(j+1)-lat(1))*a
             dUdy = (u(k,j+1,i)-u(k,j,i))/(dy)
           else if( j .eq. ny ) then
             dy =(lat(j)-lat(j-1))*a
             dUdy = (u(k,j,i)-u(k,j-1,i))/(dy)
           else
             dy =(lat(j+1)-lat(j-1))*a
             dUdy = (u(k,j+1,i)-u(k,j-1,i))/(dy)
           end if
           Z(k,j,i)=dVdx-dUdy
          end do
        end do
      end do
      return
      end

      subroutine dthdp(dT,TH,p,nt,nz,ny,nx)
      implicit none
!f2py intent(out) dT
!f2py intent(in) TH,p,lat,lon,nt,nz,ny,nx
      integer nt,nz,ny,nx
      real dT(nt,nz,ny,nx)
      real TH(nz)
      real p(nt,nz,ny,nx)
      integer n,k,j,i
      do n=1,nt
        do k=2,nz-1
          do j=1,ny
            do i=1,nx
              dT(n,k,j,i)=(TH(k+1)-TH(k-1))/
     $                    (p(n,k+1,j,i)-p(n,k-1,j,i))
            end do
          end do
        end do
      end do
      return
      end

      subroutine dthdp3d(dT,TH,p,nz,ny,nx)
      implicit none
!f2py intent(out) dT
!f2py intent(in) TH,p,lat,lon,nt,nz,ny,nx
      integer nz,ny,nx
      real dT(nz,ny,nx)
      real TH(nz)
      real p(nz,ny,nx)
      integer k,j,i
      do k=2,nz-1
        do j=1,ny
          do i=1,nx
            dT(k,j,i)=(TH(k+1)-TH(k-1))/
     $                  (p(k+1,j,i)-p(k-1,j,i))
          end do
        end do
      end do
      return
      end
