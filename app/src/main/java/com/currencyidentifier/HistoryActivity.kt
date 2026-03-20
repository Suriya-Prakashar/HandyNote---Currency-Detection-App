package com.currencyidentifier

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import org.json.JSONArray
import org.json.JSONObject

class HistoryActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var emptyText: TextView
    private lateinit var totalText: TextView
    private lateinit var adapter: HistoryAdapter
    private val historyList = mutableListOf<HistoryItem>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        recyclerView = findViewById(R.id.historyRecyclerView)
        emptyText = findViewById(R.id.emptyHistoryText)
        totalText = findViewById(R.id.totalValueText)
        findViewById<ImageButton>(R.id.btnBack).setOnClickListener { finish() }

        loadHistory()
        setupRecyclerView()
        updateTotal()
    }

    private fun loadHistory() {
        val prefs = getSharedPreferences("currency_history", Context.MODE_PRIVATE)
        val jsonString = prefs.getString("history_data", "[]") ?: "[]"
        val jsonArray = JSONArray(jsonString)
        
        historyList.clear()
        for (i in 0 until jsonArray.length()) {
            val obj = jsonArray.getJSONObject(i)
            historyList.add(HistoryItem(
                obj.getLong("id"),
                obj.getString("amount"),
                obj.getString("date")
            ))
        }
        historyList.sortByDescending { it.id }
    }

    private fun setupRecyclerView() {
        adapter = HistoryAdapter(historyList) { item ->
            deleteItem(item)
        }
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter
        checkEmptyState()
    }

    private fun deleteItem(item: HistoryItem) {
        historyList.remove(item)
        saveHistory()
        adapter.notifyDataSetChanged()
        updateTotal()
        checkEmptyState()
    }

    private fun saveHistory() {
        val prefs = getSharedPreferences("currency_history", Context.MODE_PRIVATE)
        val jsonArray = JSONArray()
        for (item in historyList) {
            val obj = JSONObject()
            obj.put("id", item.id)
            obj.put("amount", item.amount)
            obj.put("date", item.date)
            jsonArray.put(obj)
        }
        prefs.edit().putString("history_data", jsonArray.toString()).apply()
    }

    private fun updateTotal() {
        var total = 0
        for (item in historyList) {
            val value = item.amount.replace("₹", "").trim().toIntOrNull() ?: 0
            total += value
        }
        totalText.text = "Total: ₹$total"
    }

    private fun checkEmptyState() {
        if (historyList.isEmpty()) {
            emptyText.visibility = View.VISIBLE
            recyclerView.visibility = View.GONE
        } else {
            emptyText.visibility = View.GONE
            recyclerView.visibility = View.VISIBLE
        }
    }

    data class HistoryItem(val id: Long, val amount: String, val date: String)

    class HistoryAdapter(
        private val items: List<HistoryItem>,
        private val onDelete: (HistoryItem) -> Unit
    ) : RecyclerView.Adapter<HistoryAdapter.ViewHolder>() {

        class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
            val amountText: TextView = view.findViewById(R.id.amountText)
            val dateText: TextView = view.findViewById(R.id.dateText)
            val btnDelete: ImageButton = view.findViewById(R.id.btnDelete)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val view = LayoutInflater.from(parent.context).inflate(R.layout.item_history, parent, false)
            return ViewHolder(view)
        }

        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            val item = items[position]
            holder.amountText.text = item.amount
            holder.dateText.text = item.date
            holder.btnDelete.setOnClickListener { onDelete(item) }
        }

        override fun getItemCount() = items.size
    }
}
