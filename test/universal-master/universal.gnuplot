set terminal pslatex auxfile
set size 1.0,0.9
set output "universal.tex"
set format cb "%3.1f"
unset key
set view map
set style data pm3d
#set style function pm3d
set isosamples 100
set palette defined ( 0 "green", 1 "yellow", 2 "red" )
set noztics
set cbtics 2.0 norangelimit
set xrange [ 0.5 : 399.5 ] noreverse nowriteback
set yrange [ 0.5 : 239.5 ] noreverse nowriteback
set xlabel "$\\log_{10}(h/T)$"
set noxlabel
set ylabel "$\\log_{10} (1 - e)$"
set xtics('$-3$' 0.5, '$-2$' 133.33, '$-1$' 266.66, '$0$' 399.5)
set xtics('$\ $' 133.33, '$\ $' 266.66)
set ytics('$\mbox{\em}0$' 0.5, '$\mbox{\em}-2$' 60.0, '$\mbox{\em}-4$' 120.0, '$\mbox{\em}-6$' 180.0, '$\mbox{\em}-8$' 239.5)
set cbrange [ -16.0 : -6.0 ] noreverse nowriteback
set pm3d map
plot 'universal.dat' matrix with image
set output
