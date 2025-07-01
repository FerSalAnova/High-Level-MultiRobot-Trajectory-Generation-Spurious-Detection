import java.io.*;
import java.util.*;

public class SpecNetGenerator {

    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: java AutomatonToPNML <input_file>");
            return;
        }

        String inputFilename = args[0];
        String outputFilename = "output.pnml";

        try {
            BufferedReader reader = new BufferedReader(new FileReader(inputFilename));
            List<String> places = new ArrayList<>();
            Set<String> initialState = new HashSet<>();
            Set<String> acceptingStates = new HashSet<>();
            Map<String, Map<String, List<String>>> transitions = new HashMap<>();

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("States:")) {
                    String statesLine = line.substring("States: ".length()).trim();
                    statesLine = statesLine.replaceAll("\\[|\\]", "");
                    places.addAll(Arrays.asList(statesLine.split(", ")));
                } else if (line.startsWith("Initial State:")) {
                    initialState.add(line.substring("Initial State: ".length()).trim());
                } else if (line.startsWith("Accepting States:")) {
                    String acceptingLine = line.substring("Accepting States: ".length()).trim();
                    acceptingLine = acceptingLine.replaceAll("\\[|\\]", "");
                    acceptingStates.addAll(Arrays.asList(acceptingLine.split(", ")));
                } else if (line.startsWith("Transitions=")) {
                    String transitionsLine = line.substring("Transitions=".length()).trim();
                    transitions = parseTransitions(transitionsLine);
                }
            }
            reader.close();

            generatePNML(places, initialState, acceptingStates, transitions, outputFilename);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Map<String, Map<String, List<String>>> parseTransitions(String transitionsLine) {
        Map<String, Map<String, List<String>>> transitions = new HashMap<>();

        // Remove unnecessary spaces and outer curly brackets
        transitionsLine = transitionsLine.replaceAll("\\s+", "");
        transitionsLine = transitionsLine.replaceAll("^\\{", "").replaceAll("\\}$", "");

        System.out.println("Processed transitions line: " + transitionsLine); // Debug

        // Split by commas between each transition block
        String[] blocks = transitionsLine.split("},");
        System.out.println("Split into blocks: ");

        for (String block : blocks) {
            block = block + "}";
            System.out.println("Block: " + block); // Debug

            // Remove the curly brackets from the start and end of each block
            block = block.replaceAll("[{}]", "").trim();


            // Split the block into 'fromPlace' and its transitions part
            String[] parts = block.split("=", 2); // Only split at the first '='
            if (parts.length == 2) {
                String fromPlace = parts[0].trim(); // Extract the 'fromPlace'
                System.out.println("From Place: " + fromPlace); // Debug

                // Now extract the transitions part (the second part after "=")
                String transitionPart = parts[1].trim();
                System.out.println("Transition Part: " + transitionPart); // Debug

                // Further split the transitions part by commas, ensuring we are handling cases with '=' inside
                String[] transitionParts = transitionPart.split(",");
                Map<String, List<String>> toStateConditions = new HashMap<>();

                // Process each transition
                for (String transition : transitionParts) {
                    // Split each transition into toPlace and conditions
                    String[] conditionParts = transition.split("=", 2); // Only split at the first '='

                    if (conditionParts.length == 2) {
                        String toPlace = conditionParts[0].trim();
                        String conditions = conditionParts[1].replaceAll("[\\[\\]]", "").trim(); // Remove brackets

                        // Debugging the toPlace and conditions
                        System.out.println("  To Place: " + toPlace + " | Conditions: " + conditions); // Debug

                        // Split conditions by '&&' into a list
                        List<String> conditionsList = Arrays.asList(conditions.split("&&"));
                        toStateConditions.put(toPlace, conditionsList);
                    } else {
                        System.out.println("Skipping invalid transition: " + transition); // Debug
                    }
                }

                // Put the fromPlace and its transitions in the map
                transitions.put(fromPlace, toStateConditions);
            } else {
                System.out.println("Skipping invalid block: " + block); // Debug
            }
        }

        return transitions;
    }


    public static void generatePNML(
            List<String> places,
            Set<String> initialState,
            Set<String> acceptingStates,
            Map<String, Map<String, List<String>>> transitions,
            String outputFilename) {

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilename))) {
            writer.write("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n");
            writer.write("<pnml xmlns=\"http://www.pnml.org/version-2009/grammar/pnml\">\n  <net id=\"buchi_automaton\" type=\"RefNet\">\n");

            // Generate places with formatted structure
            int placeId = 1;
            Map<String, String> placeMap = new HashMap<>();
            for (String place : places) {
                String placeTag = "p" + placeId;
                placeMap.put(place, placeTag);
                writer.write("    <place id=\"" + placeTag + "\">\n");

                // Add initial marking if applicable
                if (initialState.contains(place)) {
                    writer.write("      <initialMarking>\n");
                    writer.write("         <text>[]</text>\n");
                    writer.write("      </initialMarking>\n");
                }

                // Add place name with graphics offset
                writer.write("      <name>\n");
                writer.write("         <graphics>\n");
                writer.write("            <offset x=\"15\" y=\"15\"/>\n");
                writer.write("         </graphics>\n");
                writer.write("         <text>" + place + "</text>\n");
                writer.write("      </name>\n");

                writer.write("    </place>\n");
                placeId++;
            }

            // Generate transitions and arcs
            int transitionId = placeId;
            int arcId = 0;
            for (Map.Entry<String, Map<String, List<String>>> transition : transitions.entrySet()) {
                String fromPlace = placeMap.get(transition.getKey());

                for (Map.Entry<String, List<String>> toState : transition.getValue().entrySet()) {
                    String toPlace = placeMap.get(toState.getKey());

                    // Get the conditions directly
                    List<String> conditions = toState.getValue();
                    List<String> conditionStringList = processConditions(conditions, transitionId);

                    // Create transitions for each condition
                    for (String condition : conditionStringList) {
                        String transitionTag = "t" + transitionId++; // Increment transitionId for each separate condition

                        // Write the transition
                        writer.write("    <transition id=\"" + transitionTag + "\">\n");
                        writer.write("      <graphics>\n");
                        writer.write("         <position x=\"0\" y=\"0\"/>\n");
                        writer.write("         <dimension x=\"20\" y=\"20\"/>\n");
                        writer.write("         <fill color=\"rgb(112,219,147)\"/>\n");
                        writer.write("         <line color=\"rgb(0,0,0)\"/>\n");
                        writer.write("      </graphics>\n");
                        writer.write("      <uplink>\n");
                        writer.write("         <graphics>\n");
                        writer.write("            <offset x=\"0\" y=\"-18\"/>\n");
                        writer.write("         </graphics>\n");

                        // Print the condition for debugging purposes
                        System.out.println("Adding condition for transition " + transitionTag + ": " + condition);

                        writer.write("         <text>:b(\"" + condition + "\")</text>\n");
                        writer.write("      </uplink>\n    </transition>\n");

                        // Define arcs
                        writer.write("    <arc id=\"a" + (arcId++) + "\" source=\"" + fromPlace + "\" target=\"" + transitionTag + "\"/>\n");
                        writer.write("    <arc id=\"a" + (arcId++) + "\" source=\"" + transitionTag + "\" target=\"" + toPlace + "\"/>\n");
                    }
                }
            }

            // Add a special transition for each accepting state with :end() text
            for (String acceptingState : acceptingStates) {
                String endTransitionTag = "t" + (transitionId++);
                writer.write("    <transition id=\"" + endTransitionTag + "\">\n");
                writer.write("      <graphics>\n");
                writer.write("         <position x=\"0\" y=\"0\"/>\n");
                writer.write("         <dimension x=\"20\" y=\"20\"/>\n");
                writer.write("         <fill color=\"rgb(219,112,219)\"/>\n"); // Different color for end transition
                writer.write("         <line color=\"rgb(0,0,0)\"/>\n");
                writer.write("      </graphics>\n");
                writer.write("      <uplink>\n");
                writer.write("         <graphics>\n");
                writer.write("            <offset x=\"0\" y=\"-18\"/>\n");
                writer.write("         </graphics>\n");
                writer.write("         <text>:end()</text>\n"); // Add :end() text
                writer.write("      </uplink>\n    </transition>\n");

                // Define arc from the accepting state to the end transition
                String acceptingPlace = placeMap.get(acceptingState);
                writer.write("    <arc id=\"a" + (arcId++) + "\" source=\"" + acceptingPlace + "\" target=\"" + endTransitionTag + "\"/>\n");
            }

            writer.write("  </net>\n</pnml>");

            System.out.println("PNML file successfully generated: " + outputFilename);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Process conditions directly from parseTransition's output
    private static List<String> processConditions(List<String> conditions, int transitionId) {
        List<String> processedConditions = new ArrayList<>();

        // Here, we combine conditions with && into a single string, and split the || conditions
        String combinedCondition = String.join(",", conditions);  // Combine && conditions

        // Check if there are || conditions
        String[] orConditions = combinedCondition.split("\\|\\|");
        if (orConditions.length > 1) {
            // If there are multiple OR conditions, each one will become a separate transition
            for (String orCondition : orConditions) {
                processedConditions.add(orCondition.trim());
            }
        } else {
            // Otherwise, treat all conditions as a single AND transition
            processedConditions.add(combinedCondition);
        }

        return processedConditions;
    }



}












