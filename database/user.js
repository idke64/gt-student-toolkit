const mongoose = require('mongoose')

const messageSchema = new mongoose.Schema({
	content: {
		type: String,
		required: true
	},
	sender: {
		type: String,
		enum: ['user', 'chatbot'],
		required: true
	},
	timestamp: {
		type: Date,
		default: () => Date.now()
	}
})

const userSchema = new mongoose.Schema({
	email: {
		type: String,
		required: true,
		lowercase: true,
		unique: true // if there are duplicate in the database then mongodb can't make indices for unique emails -- have to sync them
	},
	name: {
		type: String,
		required: true
	},
	date_created: {
		type: Date,
		default: () => Date.now()
	},
	messages: [messageSchema]
})

module.exports = mongoose.model("User", userSchema)