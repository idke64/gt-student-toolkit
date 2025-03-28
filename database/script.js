const mongoose = require('mongoose')
const user = require("./user")

mongoose.connect("mongodb://localhost/toolkit", () => {
	console.log("mongodb database connected.")
}, e => console.log(e))

async function addMessageToUser(userId, messageContent, sender) {
	try {
		const user = await User.findById(userId);

		if (!user) {
			console.log('User not found');
			return;
		}

		user.messages.push({
			content: messageContent,
			sender: sender, // Should be 'user' or 'chatbot'
		});

		await user.save();
		console.log('Message added successfully:', user);
	} catch (error) {
		console.error('Error adding message:', error);
	}
}

async function run() {
	const u = await user.create({email: "test@test.com",name:"test"})
	console.log(u)
}
run()