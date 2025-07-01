// Code for the "Performance" class, which computes sinformation about the solutions
// obtained after simulation in the paper
// "Multi-robot Motion Planning based on Nets-within-Nets Modeling and Simulation"
// Sofia Hustiu, Eva Robillard, Joaquín Ezpeleta, Cristian Mahulea and Marius Kloetzer
// To compile all the required files, just execute "javac *.java"
// Since it is just analyzing the solutions in the resulting "log.txt" file, it can be
// executed off-line, and extended with anu other performance indexes that could be considered

// It takes as input parameter the name of log file as saved by the experimental renew model, let's say "log.txt",
// with solutions of the form 
//      r4(5.2):g-r6(5.2):i
//      r7(3.5):d-r6(5.2):w3

//      r6(3.5):h-r7(6):w3-r3(5.2):w2-r2(3.5):d-r5(5.2):i
//      r7(6):l

//      .....
// computes some indexes of the solutions, and save the indexes
// in a file named "log_indexes.txt", in the form
//      min:      79.5
//      max:      780.0000000000002
//      mean:     298.377
//      maxSteps: 111
//      minSteps: 9
//      minPlan:  
//      ....

import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.FileWriter;

public class Perf_eval {
    static private String filename;
    //---------------------------------------------------------------------------
    // Initialization function with initial global capacity, region occupancy
    // and h function
    public Perf_eval(String file) {
        filename = file;
    }
    //---------------------------------------------------------------------------
    //given an input of de form "r1(3.5):b1-r3(5.0):b3" returns an array
    //with {3.5,5.0}
    private static ArrayList<Double> get_times(String t) {
        ArrayList<Double> values = new ArrayList<>();

        // numbers can be of the forms "34" or "23.678"
        Pattern pattern = Pattern.compile("\\((\\d+(\\.\\d+)?)\\)");
        Matcher matcher = pattern.matcher(t);

        while (matcher.find()) {
            String match = matcher.group();
            values.add(Double.valueOf(match.substring(1, match.length() - 1)));
        }

        return values;
    }
    // returns the sum of the elements in "values"
    private static double sum_vals(ArrayList<Double> values) {
        double res = 0.0;

        for(double v : values) {
            res += v;
        }
        return res;
    }

    //values must have at least one element
    //returns the max in "values"
    private static double max_value(ArrayList<Double> values) {
        double res = values.get(0);

        for(int i=1; i<values.size(); i++) {
            res = res<values.get(i)?values.get(i):res;
        }
        return res;
    }

    // "plan" is of the form
    //     r2(3):b1\nr1(3):b2-r3(5):b3-r2(3):b4
    // Since each line corresponds to a (synchronized) step of multiple robots,
    // the execution time of the step is computed as the maximun of its elements.
    // The total execution time will be the sum of the times of all the steps
    // For the example, the result is 8(=3+5)
    private static double compute_time(String plan) {
        double totalTime = 0.0;
        String steps[] = plan.split("\n");

        for(String step : steps) {
            totalTime += max_value(get_times(step));
        }
        return totalTime;
    }

    // Compute the number of robo moves of a complete plan
    // Basically, it is the number of times ":" appears
    private static long compute_robot_moves(String plan) {
        //Java8: counts the number of '\n' in a string
        return plan.chars().filter(nl -> nl == ':').count();
    }

    // The one requiring less execution time
    // We assume both parameters are not emtpy strings
    private static String select_best_time(String oldV, String newV) {
        double oldTime = compute_time(oldV);
        double newTime = compute_time(newV);

        if (oldTime > newTime) {//initial case or new is longer
           return newV;
        }
        else {
           return oldV;
        }
    }

    // Given two solution paths, returns the "best" one. In this version,
    // "best" means less steps in the mission net (in a solution, steps are
    // in the string separated by means of '\n')
    public static String select_best_num_steps(String oldV, String newV) {
        //Java8: counts the number of '\n' in a string
        long oldVc = oldV.chars().filter(nl -> nl == '\n').count();
        long newVc = newV.chars().filter(nl -> nl == '\n').count();
        if ((oldVc == 0) || (oldVc > newVc)) {//initial case or new is longer
           return newV;
        }
        else {
           return oldV;
        }
    }

    public static String select_best(String oldV, String newV) {
        if(oldV.length() == 0) {
            return newV;
        }
        else {
            return select_best_time(oldV,newV);
        }
    }
    //---------------------------------------------------------------------------
    public static void post_process() {
        //java 11
        String content = "";
        try {
            content = Files.readString(Paths.get(filename));
        }
        catch(Exception e) {
            System.out.println("Problems with file " + filename);
        }

        int posLastDot = filename.lastIndexOf(".");
        String baseName = filename.substring(0, posLastDot);
        String extension = filename.substring(posLastDot + 1,filename.length());
        String outputFileName = baseName + "_indexes." + extension;
        String[] plans = content.split("\n\n");

        ArrayList<Double> times = new ArrayList<Double>();
        //solutions with less and more steps and robot movements
        String maxStepsP, minStepsP;
        long minSteps, maxSteps;
        String minRobotMovesP, maxRobotMovesP;
        long minRobotMoves, maxRobotMoves;

        long len = plans[0].split("\n").length;
        long numMoves = compute_robot_moves(plans[0]);
        minSteps = len; //steps of the shortest solution
        maxSteps = len; //steps fo the longest solution
        maxStepsP = plans[0];
        minStepsP = plans[0];
        minRobotMoves = numMoves;
        maxRobotMoves = numMoves;
        minRobotMovesP = plans[0];
        maxRobotMovesP = plans[0];
        times.add(compute_time(plans[0]));

        for(int i=1; i<plans.length; i++) {
            times.add(compute_time(plans[i]));
            len = plans[i].split("\n").length;
            if(maxSteps<len) {
                maxSteps = len;
                maxStepsP = plans[i];
            }
            if(minSteps>len) {
                minSteps = len;
                minStepsP = plans[i];
            }
            numMoves = compute_robot_moves(plans[i]);
            if(maxRobotMoves<numMoves) {
                maxRobotMoves = numMoves;
                maxRobotMovesP = plans[i];
            }
            if(minRobotMoves>numMoves) {
                minRobotMoves = numMoves;
                minRobotMovesP = plans[i];
            }
        }

        //solutions requiering more/less time
        String maxP, minP;
        //min, max and mean times of all the found solutions
        double min, max, mean;
        
        min = times.get(0);
        max = times.get(0);
        mean = times.get(0);
        maxP = plans[0];
        minP = plans[0];
        mean = min / times.size();

        for(int i=1; i<times.size(); i++) {
            if(max<times.get(i)) {
                max = times.get(i);
                maxP = plans[i];
            }
            if(min>times.get(i)) {
                min = times.get(i);
                minP = plans[i];
            }
            mean += times.get(i);
        }

        mean = mean / times.size();
        
        String info = "min time:      " + min + "\n" +
                      "max time:      " + max + "\n" +
                      "mean time:     " + mean + "\n" +
                      "minSteps:      " + minSteps + "\n" +
                      "maxSteps:      " + maxSteps + "\n" +
                      "minRobotMoves: " + minRobotMoves + "\n" +
                      "maxRobotMoves: " + maxRobotMoves + "\n" +
                      "minPlan:  \n" + minP + "\n" +
                      "minSteps: \n" + minP + "\n" + 
                      "minRobotMovesP: \n" + minRobotMovesP + "\n" +
                      "----------------------------------------\n";
        // System.out.println(info);
        try {
            FileWriter fw = new FileWriter(outputFileName,true); //add results new experiment
            fw.write(info);
            fw.close();
        }
        catch(Exception e) {
            System.out.println("Problems with file " + outputFileName);
        } 
    }
    //---------------------------------------------------------------------------
    //For testing
    public static void main(String[] args) {
        if (args.length > 0) {
            Perf_eval per = new Perf_eval(args[0]);
            per.post_process();
        }
        else {
            System.err.println("Use as : 'java Perf_eval <log_file>'");
        }
    }
}