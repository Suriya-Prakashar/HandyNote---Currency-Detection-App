package com.currencyidentifier;

import android.content.Context;
import android.widget.Toast;

/**
 * Java utility class demonstrating Java+Kotlin interop
 * This class can be called from Kotlin code seamlessly
 */
public class CurrencyHelper {
    
    private static final String TAG = "CurrencyHelper";
    
    /**
     * Format currency result string (Java method callable from Kotlin)
     * 
     * @param currencyLabel The currency denomination (e.g., "100", "500")
     * @param confidence The confidence score (0.0 to 1.0)
     * @return Formatted string
     */
    public static String formatCurrencyResult(String currencyLabel, double confidence) {
        return String.format("Currency: %s\nConfidence: %.2f%%", 
            currencyLabel, confidence * 100);
    }
    
    /**
     * Show toast message (Java method callable from Kotlin)
     * 
     * @param context Android context
     * @param message Message to display
     */
    public static void showToast(Context context, String message) {
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show();
    }
    
    /**
     * Validate currency label
     * 
     * @param label Currency label to validate
     * @return true if valid Indian currency denomination
     */
    public static boolean isValidCurrencyLabel(String label) {
        if (label == null || label.isEmpty()) {
            return false;
        }
        // Valid Indian currency denominations
        String[] validLabels = {"5", "10", "20", "50", "100", "200", "500", "2000"};
        for (String valid : validLabels) {
            if (valid.equals(label)) {
                return true;
            }
        }
        return false;
    }
}
