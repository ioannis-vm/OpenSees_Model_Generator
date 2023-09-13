numberer {%NUMBERER%}
constraints {%CONSTRAINTS%}
system {%SYSTEM%}

{%DAMPING CODE%}

test EnergyIncr 1.0e-6 100 0
integrator TRBDF2
algorithm KrylovNewton
analysis Transient

set output_dir {%output_dir%}
set analysis_time_increment {%time_increment%}

set drift_nodes {{%drift_nodes%}}
set drift_heights {{%drift_heights%}}

# initialize loop variables
set num_subdiv 0
set num_times 0
set total_step_count 0
set analysis_failed 0
set curr_time 0.00

set target_timestamp {%target_timestamp%}

set skip_steps {%skip_steps%}

set scale {
    1.0 1.0e-1 1.0e-2 1.0e-3
    1.0e-4 1.0e-5 1.0e-6 1.0e-7
    1.0e-8 1.0e-9 1.0e-10
}
set tols {
    1.0e-8 1.0e-8 1.0e-8 1.0e-8 1.0e-8 
    1.0e-8 1.0e-8 1.0e-8 1.0e-8 1.0e-8
    1.0e-8
}

set start_time [clock seconds]
set the_time $start_time


proc writeToFile {string filename} {
   # Open the file in write mode
   set file [open $filename "w"]

   # Write the string to the file
   puts $file $string

   # Close the file
   close $file
}

writeToFile "running" $output_dir/status

while {[expr $curr_time + 1e-7] < $target_timestamp} {
    if {$analysis_failed} {
        break
    }

    test "EnergyIncr" [lindex $tols $num_subdiv] 200 3 2
    set check [analyze 1 [expr {$analysis_time_increment * [lindex $scale $num_subdiv]}]]
    incr total_step_count

    if {$check != 0} {
        # analysis failed
        if {$num_subdiv == [llength $scale] - 1} {
            # can't subdivide any further
            puts "==========================="
            puts "Analysis failed to converge"
            puts "==========================="
            set analysis_failed 1
	    writeToFile "Analysis failed to converge" $output_dir/status
            break
        }

        # otherwise, we can still reduce step size
        incr num_subdiv
        # how many times to run with reduced step size
        set num_times 50
    } else {
        # analysis was successful
        set prev_time $curr_time
        set curr_time [expr {[getTime]*1.0}]

        # log entry for analysis status
        if {[clock seconds] - $the_time > 10.0} {
            set the_time [clock seconds]
            # total time running
            set running_time [expr {$the_time - $start_time}]
            # the seconds ran is `curr_time`
            set remaining_time [expr {$target_timestamp - $curr_time}]
            set average_speed [expr {$curr_time / $running_time}]  ;# th [s] / real [s]
            # estimated remaining real time to finish [s]
            set est_remaining_dur [expr {$remaining_time / $average_speed}]
	    writeToFile [concat "running: " [expr {round($est_remaining_dur)}] "sec remaining."] $output_dir/status
        }

	# drift check
	for {set i 0} {$i < [expr [llength $drift_nodes] - 1]} {incr i} {
	    set n_down [lindex $drift_nodes $i]
	    set n_up [lindex $drift_nodes [expr $i + 1]]
	    set height [lindex $drift_heights $i]
	    for {set dir 1} {$dir <= 3} {incr dir} {
		set disp_down [nodeDisp $n_down $dir]
		set disp_up [nodeDisp $n_up $dir]
		set drift [expr ($disp_up - $disp_down)/$height]
		if {$drift > {%collapse_drift%}} {
		    puts "========================================"
		    puts "Analysis stopped due to excessive drift."
		    puts "========================================"
		    writeToFile "collapse" $output_dir/status
		    set analysis_failed 1
		    break
		}
	    }
	}

        if {$num_times != 0} {
            incr num_times -1
        }

        if {$total_step_count % $skip_steps == 0} {
            set n_steps_success [incr n_steps_success]
            lappend time_vector $curr_time
        }

        if {$num_subdiv != 0} {
            if {$num_times == 0} {
                incr num_subdiv -1
                set num_times 50
            }
        }
    }
}

if {$analysis_failed != 1} {
    writeToFile "Analysis finished" $output_dir/status
}

