import java.io.*;
import java.util.*;
import java.util.regex.*;

public class LTLBuchiAutomaton{

    // Function to preprocess the LTL formula to ensure it's compatible with ltl2ba
    public static String preprocessFormula(String formula) {
        /*formula = formula.replaceAll("[pPoOY]", "y");
        formula = formula.replaceAll("[^y0-9()<>\\-!&|FGU]", "");
        formula = formula.replaceAll("U", " U ");
        formula = formula.replaceAll("R", " V ");
        formula = formula.replaceAll("F", " <> ");
        formula = formula.replaceAll("G", " [] ");*/

        formula = formula.replaceAll("\\bU\\b", " U ");
        formula = formula.replaceAll("\\bR\\b", " V ");
        formula = formula.replaceAll("\\bF\\b", " <> ");
        formula = formula.replaceAll("\\bG\\b", " [] ");

        // Clean up extra whitespace
        formula = formula.replaceAll("\\s+", " ").trim();

        return formula.replaceAll("  ", " ");
    }

    // Function to execute the ltl2ba tool and capture its output
    public static String runLTL2BA(String formula) throws IOException {
        String os = System.getProperty("os.name").toLowerCase();
        String ltl2baPath = "./aux_toolboxes/ltl2ba/ltl2ba"; // Path to ltl2ba executable
        if (os.contains("win")) {
            ltl2baPath += ".exe"; // For Windows users
        }

        ProcessBuilder processBuilder = new ProcessBuilder(ltl2baPath, "-d", "-f", formula);
        processBuilder.redirectErrorStream(true);
        Process process = processBuilder.start();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;

        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }

        try {
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                System.err.println("ltl2ba failed with exit code: " + exitCode);
            }
        } catch (InterruptedException e) {
            System.err.println("Process was interrupted: " + e.getMessage());
            Thread.currentThread().interrupt();
        }

        return output.toString();
    }

    // Function to extract the 'never' block from the captured output
    public static String extractNeverBlock(String output) {
        Pattern pattern = Pattern.compile("never \\{(.*?)\\}", Pattern.DOTALL);
        Matcher matcher = pattern.matcher(output);

        if (matcher.find()) {
            // Extract the 'never' block
            String neverBlock = matcher.group(1).trim();

            // Replace all occurrences of accept_* with accept_all
            neverBlock = neverBlock.replaceAll("accept_\\S+", "accept_all");

            // Split the block by lines and process each block
            String[] blocks = neverBlock.split("\n(?=T\\S+:|accept_all:)"); // Split into individual state blocks
            StringBuilder nonAcceptStates = new StringBuilder();

            for (String block : blocks) {
                String stateName = block.split(":")[0].trim(); // Extract state name by splitting at ':'

                if (!stateName.equals("accept_all")) {
                    // If it's not an acceptance state, keep the block as is
                    nonAcceptStates.append(block).append("\n");
                }
            }

            // Append the single accept_all state at the end
            nonAcceptStates.append("accept_all: \n    skip\n");

            // Return the modified 'never' block with all accept states removed and a single accept_all block added at the end
            return "never {\n" + nonAcceptStates.toString().trim() + "\n}";
        } else {
            return "No 'never' block found!";
        }
    }

    public static void parseNever(String neverBlock, String outputFilePath) throws IOException {
        // Regular expressions to match state definitions and transitions
        Pattern statePattern = Pattern.compile("\\bT\\S+:");  // Match all states ending with ":"
        Pattern transitionPattern = Pattern.compile("::\\s*\\((.*?)\\)\\s*->\\s*goto\\s*(T\\S+|accept_\\d+|accept_all)");

        // Maps to store states and transitions
        Set<String> states = new LinkedHashSet<>();  // Use LinkedHashSet to preserve order and avoid duplicates
        Map<String, Map<String, List<String>>> transitions = new HashMap<>();
        Set<String> acceptStates = new HashSet<>();
        String initialState = null;

        // Match all states in the never block (e.g., T0_init, T0_S4)
        Matcher stateMatcher = statePattern.matcher(neverBlock);
        while (stateMatcher.find()) {
            String state = stateMatcher.group().replace(":", "").trim();  // Remove the colon and trim
            states.add(state);  // Automatically handles duplicates and maintains order
            // Set the first found state as initial state (assuming the first state is the initial state)
            if (initialState == null) {
                initialState = state;
            }
        }

        // Match transitions for each state
        String[] blocks = neverBlock.split("\\n(?=T\\S+:)"); // Split into individual state blocks (each block contains transitions)
        for (String block : blocks) {
            // Extract the state name from the block
            String stateName = block.split(":")[0].trim(); // Extract state name by splitting at ':'
            if (!states.contains(stateName)) continue;

            // Find all transitions in the block
            Matcher transitionMatcher = transitionPattern.matcher(block);
            while (transitionMatcher.find()) {
                String condition = transitionMatcher.group(1).trim();
                String targetState = transitionMatcher.group(2).trim();

                // If the condition contains "||" (logical OR), we need to fix the parentheses and reformat it
                if (condition.contains("||")) {
                    // Fix parentheses and split conditions on "||"
                    condition = condition.replaceAll("[\\(\\)]", "");  // Remove all parentheses
                    String[] subConditions = condition.split("\\s*\\|\\|\\s*");  // Split by "||"
                    condition = String.join(" || ", subConditions);  // Rejoin them with " || "
                }

                // Now, add the condition (formatted) to the transitions map
                transitions.computeIfAbsent(stateName, k -> new HashMap<>());
                transitions.get(stateName).computeIfAbsent(targetState, k -> new ArrayList<>()).add(condition);

                // If target state is accepting state (accept_* or accept_all)
                if (targetState.startsWith("accept_") || targetState.equals("accept_all")) {
                    acceptStates.add(targetState);
                    states.add(targetState); // Add accepting states to the States list
                }
            }
        }

        // Write the results to a file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
            // Output the States, Initial State, and Accepting States
            writer.write("States: " + states + "\n");
            writer.write("Initial State: " + initialState + "\n");
            writer.write("Accepting States: " + acceptStates + "\n");

            // Output transitions in the desired format
            writer.write("Transitions={");
            boolean first = true;
            for (Map.Entry<String, Map<String, List<String>>> entry : transitions.entrySet()) {
                String originState = entry.getKey();
                if (!first) {
                    writer.write(", ");
                }
                writer.write(originState + "={");
                boolean firstDest = true;
                for (Map.Entry<String, List<String>> trans : entry.getValue().entrySet()) {
                    String destState = trans.getKey();
                    List<String> conditions = trans.getValue();
                    if (!firstDest) {
                        writer.write(", ");
                    }
                    // Write each condition in the desired format
                    writer.write(destState + "=" + conditions);
                    firstDest = false;
                }
                writer.write("}");
                first = false;
            }
            writer.write("}\n");
        }
    }

    public static void main(String[] args) {
        try {
            Scanner scanner = new Scanner(System.in);
            System.out.print("Enter an LTL formula: ");
            String formula = scanner.nextLine();
            System.out.println("Formula entered: [" + formula + "]");

            formula = preprocessFormula(formula);
            System.out.println("Preprocessed formula: [" + formula + "]");

            String output = runLTL2BA(formula);

            String neverBlock = extractNeverBlock(output);

            System.out.println(neverBlock);

            String outputFilePath = "BuchiAutomatonOutput.txt";
            parseNever(neverBlock, outputFilePath);

            System.out.println("Output written to: " + outputFilePath);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}













