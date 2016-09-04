      subroutine interp1d(VC,X,H,NT,NZ,NY,NX,NH,X4d)
Cf2py intent(out) X4d
Cf2py intent(in) VC
Cf2py intent(in) X
Cf2py intent(in) H
Cf2py intent(in) NT,NZ,NY,NX,NH
      implicit none
      integer NT,NZ,NY,NX,NH
      real VC(NT,NZ,NY,NX)
      real X(NZ)
      real H(NH)
      real X4d(NT,NH,NY,NX)
      real h0,w1,w2
      integer hk,n,k,j,i
      integer ip,im
      logical interp
      !write(*,*) 'NT=',NT,'NZ=',NZ,'NY=',NY,'NX=',NX,'NH=',NH
      ip=0
      im=1
      if( VC(1,1,1,1) .gt. VC(1,NZ,1,1) ) then
        ip=1
        im=0
      end if
      X4d=0.0
      do hk=1,NH
        h0=H(hk)
        do n=1,NT
          do j=1,NY
            do i=1,NX
              interp=.false.
              k=NZ
              do while( (.not. interp) .and. (k .ge. 2))
                if(VC(n,k-im,j,i) .le. h0 .and. 
     1             VC(n,k-ip,j,i) .gt. h0 ) then

                  w2=(h0-VC(n,k-im,j,i))/
     1               (VC(n,k-ip,j,i)-VC(n,k-im,j,i))
                  w1=1.0-w2

                  X4d(n,hk,j,i)=w1*X(k-im)+w2*X(k-ip)
                  interp=.true.
                end if
                k=k-1
              end do
            end do
          end do
        end do
      end do
      return
      end

      subroutine interp1d3d(VC,X,H,NZ,NY,NX,NH,X3d)
Cf2py intent(out) X3d
Cf2py intent(in) VC
Cf2py intent(in) X
Cf2py intent(in) H
Cf2py intent(in) NZ,NY,NX,NH
      implicit none
      integer NZ,NY,NX,NH
      real VC(NZ,NY,NX)
      real X(NZ)
      real H(NH)
      real X3d(NH,NY,NX)
      real h0,w1,w2
      integer hk,k,j,i
      integer ip,im
      logical interp
      !write(*,*) 'NT=',NT,'NZ=',NZ,'NY=',NY,'NX=',NX,'NH=',NH
      ip=0
      im=1
      if( VC(1,1,1) .gt. VC(NZ,1,1) ) then
        ip=1
        im=0
      end if
      X3d=0.0
      do hk=1,NH
        h0=H(hk)
        do j=1,NY
          do i=1,NX
            interp=.false.
            k=NZ
            do while( (.not. interp) .and. (k .ge. 2))
              if(VC(k-im,j,i) .le. h0 .and. 
     1           VC(k-ip,j,i) .gt. h0 ) then

                w2=(h0-VC(k-im,j,i))/
     1             (VC(k-ip,j,i)-VC(k-im,j,i))
                w1=1.0-w2

                X3d(hk,j,i)=w1*X(k-im)+w2*X(k-ip)
                interp=.true.
              end if
              k=k-1
            end do
          end do
        end do
      end do
      return
      end

      subroutine interp4d(VC,X,H,NT,NZ,NY,NX,NH,X4d)
Cf2py intent(out) X4d
Cf2py intent(in) VC
Cf2py intent(in) P
Cf2py intent(in) H
      implicit none
      integer NT,NZ,NY,NX,NH
      real VC(NT,NZ,NY,NX)
      real X(NT,NZ,NY,NX)
      real H(NH)
      real X4d(NT,NH,NY,NX)
      real h0,w1,w2
      integer hk,n,k,j,i
      integer ip,im
      logical interp
      ip=0
      im=1
      if( VC(1,1,1,1) .gt. VC(1,NZ,1,1) ) then
        ip=1
        im=0
      end if
      X4d=0.0
      do hk=1,NH
        h0=H(hk)
        do n=1,NT
          do j=1,NY
            do i=1,NX
              interp=.false.
              k=NZ
              do while( (.not. interp) .and. (k .ge. 2))
                if(VC(n,k-im,j,i) .le. h0 .and. 
     1             VC(n,k-ip,j,i) .gt. h0 ) then

                  w2=(h0-VC(n,k-im,j,i))/
     1               (VC(n,k-ip,j,i)-VC(n,k-im,j,i))
                  w1=1.0-w2

                  X4d(n,hk,j,i)=w1*X(n,k-im,j,i)+w2*X(n,k-ip,j,i)
                  interp=.true.
                end if
                k=k-1
              end do
            end do
          end do
        end do
      end do
      return
      end

      subroutine interp3d(VC,X,H,NZ,NY,NX,NH,X3d)
Cf2py intent(out) X3d
Cf2py intent(in) VC
Cf2py intent(in) P
Cf2py intent(in) H
      implicit none
      integer NZ,NY,NX,NH
      real VC(NZ,NY,NX)
      real X(NZ,NY,NX)
      real H(NH)
      real X3d(NH,NY,NX)
      real h0,w1,w2
      integer hk,k,j,i
      integer ip,im
      logical interp
      ip=0
      im=1
      if( VC(1,1,1) .gt. VC(NZ,1,1) ) then
        ip=1
        im=0
      end if
      X3d=0.0
      do hk=1,NH
        h0=H(hk)
          do j=1,NY
            do i=1,NX
              interp=.false.
              k=NZ
              do while( (.not. interp) .and. (k .ge. 2))
                if(VC(k-im,j,i) .le. h0 .and. 
     1             VC(k-ip,j,i) .gt. h0 ) then

                  w2=(h0-VC(k-im,j,i))/
     1               (VC(k-ip,j,i)-VC(k-im,j,i))
                  w1=1.0-w2

                  X3d(hk,j,i)=w1*X(k-im,j,i)+w2*X(k-ip,j,i)
                  interp=.true.
                end if
                k=k-1
              end do
            end do
          end do
      end do
      return
      end
